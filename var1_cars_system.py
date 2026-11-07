"""Variant 1 (individual task): cars - contours + tracking.

video -> preprocessing -> motion (MOG2) -> road mask -> morphology -> contours -> car filter
      -> centroid tracking -> counting line -> visualisation + statistics

Usage (from the project folder, venv active):
    python var1_cars_system.py --setup      # once: click the road zone and the counting line
    python var1_cars_system.py              # run with a live window (q = quit, p = pause)
    python var1_cars_system.py --no-show    # run without a window (true FPS)
"""
from pathlib import Path
from collections import deque
import sys, json, csv, math, time
import cv2
import numpy as np

# ---------------------------------------------------------------- settings
VIDEO = Path("data") / "var1_cars.mp4"
START_S = 28.0                 # skip the start of the video (camera settling, no cars); 0 = use all
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
ROI_FILE = RESULTS / "var1_roi.json"
MAX_SIDE = 960                 # processing size: longer side in px
BLUR = 5
HISTORY, VAR_THR = 300, 25     # MOG2
WARMUP = 30                    # frames used only to learn the background (no detection)
K_OPEN, K_CLOSE = 3, 11        # morphology (elliptical kernels)
MIN_AREA, MAX_AREA = 600, 40000   # px^2 at 960 px  -> calibrate in the notebook (step 4.4)
ASPECT_MIN, ASPECT_MAX = 0.8, 5.0 # box width / height (side view: cars are wider than tall)
MIN_SOLIDITY = 0.5
MAX_DIST, MIN_HITS, MAX_MISSED, TRAIL = 80, 5, 10, 30   # tracker
SPEED_WINDOW = 10              # frames for the speed estimate

K_O = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_OPEN, K_OPEN))
K_C = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_CLOSE, K_CLOSE))


def resize(frame):
    h, w = frame.shape[:2]; s = MAX_SIDE / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def open_video():
    """Open the video and skip to START_S by reading frames (exact, unlike seeking)."""
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"Cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    for _ in range(int(round(START_S * fps))):
        cap.grab()
    return cap


def direction_names(line):
    """Readable names of the two crossing directions: left/right for a steep line, top/bottom otherwise."""
    (ax, ay), (bx, by) = line
    return ("L->R", "R->L") if abs(bx - ax) < abs(by - ay) else ("T->B", "B->T")


# ---------------------------------------------------------------- 0) setup: road zone + counting line
def setup():
    cap = open_video(); ok, frame = cap.read(); cap.release()
    if not ok:
        sys.exit("Cannot read the first frame after START_S")
    frame = resize(frame)
    poly, line, stage = [], [], ["poly"]
    win = "setup: click road corners, n = next; click 2 line points, s = save, q = quit"

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            (poly if stage[0] == "poly" else line).append((x, y))
            if stage[0] == "line" and len(line) > 2:
                line.pop(0)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.setMouseCallback(win, click)
    while True:
        vis = frame.copy()
        if poly:
            cv2.polylines(vis, [np.array(poly, np.int32)], stage[0] != "poly", (0, 255, 255), 2)
        for p in poly:
            cv2.circle(vis, p, 4, (0, 255, 255), -1)
        if len(line) == 2:
            cv2.line(vis, line[0], line[1], (0, 0, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("n") and len(poly) >= 3:
            stage[0] = "line"
        elif k == ord("s") and len(poly) >= 3 and len(line) == 2:
            ROI_FILE.write_text(json.dumps({"size": frame.shape[1::-1], "road": poly, "line": line}, indent=2))
            print("saved", ROI_FILE); break
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- 1-4) detection
def make_road_mask(shape, poly):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    return m


def detect(fg, road):
    """MOG2 output -> list of car candidates; also returns the rejection counts."""
    m = np.where(fg == 255, 255, 0).astype(np.uint8)          # drop shadows (127)
    m = cv2.bitwise_and(m, road)                              # only the road zone
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, K_O), cv2.MORPH_CLOSE, K_C)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cars, rej = [], {"small": 0, "large": 0, "aspect": 0, "solidity": 0}
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA: rej["small"] += 1; continue
        if area > MAX_AREA: rej["large"] += 1; continue
        x, y, w, h = cv2.boundingRect(c)
        if not ASPECT_MIN <= w / h <= ASPECT_MAX: rej["aspect"] += 1; continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or area / hull < MIN_SOLIDITY: rej["solidity"] += 1; continue
        cars.append(dict(x=x, y=y, w=w, h=h, area=area, cx=x + w / 2, cy=y + h / 2))
    return cars, m, rej


# ---------------------------------------------------------------- 5) tracking (Task 11 tracker + speed and line crossing)
def side(p, a, b):
    """Sign of point p relative to the line a-b (+1 / -1 / 0)."""
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return (v > 0) - (v < 0)


class CarTracker:
    def __init__(self, line):
        self.tracks, self.next_id, self.line = [], 1, line
        self.steep = direction_names(line)[0] == "L->R"
        self.crossings = []                                    # (id, frame, time_s, direction)

    def update(self, dets, frame_no, fps):
        pairs = sorted((math.hypot(t["cx"] - d["cx"], t["cy"] - d["cy"]), ti, di)
                       for ti, t in enumerate(self.tracks) for di, d in enumerate(dets))
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:                             # greedy: closest pairs first
            if dist > MAX_DIST: break
            if ti in used_t or di in used_d: continue
            used_t.add(ti); used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            t.update({k: d[k] for k in ("x", "y", "w", "h", "area", "cx", "cy")})
            t["hits"] += 1; t["missed"] = 0; t["last"] = frame_no
            t["hist"].append((frame_no, d["cx"], d["cy"])); t["trail"].append((int(d["cx"]), int(d["cy"])))
            t["areas"].append(d["area"])
            if t["id"] is None and t["hits"] >= MIN_HITS:      # confirm -> visible ID
                t["id"] = self.next_id; self.next_id += 1
            self._check_line(t, frame_no, fps)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t: t["missed"] += 1
        self.finished = [t for t in self.tracks if t["id"] is not None and t["missed"] > MAX_MISSED]
        self.tracks = [t for t in self.tracks
                       if t["missed"] <= (MAX_MISSED if t["id"] is not None else 1)]
        for di, d in enumerate(dets):
            if di not in used_d:
                t = {k: d[k] for k in ("x", "y", "w", "h", "area", "cx", "cy")}
                t.update(id=None, hits=1, missed=0, first=frame_no, last=frame_no, counted=None,
                         side=side((d["cx"], d["cy"]), *self.line),
                         hist=deque([(frame_no, d["cx"], d["cy"])], maxlen=SPEED_WINDOW),
                         trail=deque([(int(d["cx"]), int(d["cy"]))], maxlen=TRAIL),
                         areas=[d["area"]], path=0.0)
                self.tracks.append(t)
        for t in self.tracks:                                  # speed in px/s over the last frames
            (f0, x0, y0), (f1, x1, y1) = t["hist"][0], t["hist"][-1]
            t["speed"] = math.hypot(x1 - x0, y1 - y0) / ((f1 - f0) / fps) if f1 > f0 else 0.0
        return [t for t in self.tracks if t["id"] is not None and t["missed"] == 0]

    def _check_line(self, t, frame_no, fps):
        if len(t["hist"]) >= 2:
            (_, xa, ya), (_, xb, yb) = t["hist"][-2], t["hist"][-1]
            t["path"] += math.hypot(xb - xa, yb - ya)
        s = side((t["cx"], t["cy"]), *self.line)
        if s != 0 and t["side"] != 0 and s != t["side"] and t["id"] is not None and t["counted"] is None:
            (_, xa, ya), (_, xb, yb) = t["hist"][-2], t["hist"][-1]
            if self.steep:
                direction = "L->R" if xb > xa else "R->L"
            else:
                direction = "T->B" if yb > ya else "B->T"
            t["counted"] = direction
            self.crossings.append((t["id"], frame_no, round(START_S + frame_no / fps, 3), direction))
        if s != 0:
            t["side"] = s


# ---------------------------------------------------------------- 6) visualisation
def colour(i):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[(i * 37) % 180, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def text(img, s, org, col, scale=0.42):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def draw(img, cars, roi, counts, fps_now):
    d1, d2 = counts
    cv2.polylines(img, [np.array(roi["road"], np.int32)], True, (0, 255, 255), 1)
    a, b = map(tuple, roi["line"]); cv2.line(img, a, b, (0, 0, 255), 2)
    for t in cars:
        c = colour(t["id"]); x, y, w, h = t["x"], t["y"], t["w"], t["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        pts = list(t["trail"])
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, c, 2)
        lines = [f"Car ID: {t['id']}", f"X: {x}  Y: {y}", f"W: {w}  H: {h}",
                 f"Area: {int(t['area'])}", f"Speed: {t['speed']:.0f} px/s"]
        for k, s in enumerate(lines):
            text(img, s, (x, max(12, y - 6 - 13 * (len(lines) - 1 - k))), c)
    areas = [t["area"] for t in cars]
    panel = [f"Cars in frame: {len(cars)}",
             f"Counted: {counts[d1] + counts[d2]}  ({d1} {counts[d1]}, {d2} {counts[d2]})",
             "Area mean/max/min: " + (f"{np.mean(areas):.0f} / {max(areas):.0f} / {min(areas):.0f}" if areas else "-"),
             f"FPS: {fps_now:.1f}"]
    cv2.rectangle(img, (0, 0), (330, 16 + 18 * len(panel)), (40, 40, 40), -1)
    for k, s in enumerate(panel):
        text(img, s, (8, 18 + 18 * k), (255, 255, 255), 0.5)


# ---------------------------------------------------------------- main loop
def main(show=True, tag="", **override):
    """Run the system; keyword overrides change settings for this run only, e.g. main(show=False, MIN_AREA=900)."""
    g = globals()
    old = {k: g[k] for k in override}
    g.update(override)
    try:
        return _run(show, tag)
    finally:
        g.update(old)                                          # restore the defaults


def _run(show, tag):
    if not ROI_FILE.exists():
        sys.exit("Run with --setup first to mark the road zone and the counting line")
    roi = json.loads(ROI_FILE.read_text())
    cap = open_video(); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = cap.read()
    if not ok:
        sys.exit("Cannot read the video")
    small = resize(frame); H, W = small.shape[:2]
    road = make_road_mask(small.shape, roi["road"])
    mog = cv2.createBackgroundSubtractorMOG2(history=HISTORY, varThreshold=VAR_THR, detectShadows=True)
    tracker = CarTracker([tuple(p) for p in roi["line"]])
    out = cv2.VideoWriter(str(RESULTS / f"var1_output{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2, H))
    frames_rows, obj_rows, all_tracks = [], [], {}
    rej_total = {"small": 0, "large": 0, "aspect": 0, "solidity": 0}
    DIRS = direction_names(roi["line"])
    counts = {d: 0 for d in DIRS}
    times, n, t_loop = deque(maxlen=30), 0, time.perf_counter()
    while ok:
        t0 = time.perf_counter()
        small = resize(frame)
        fg = mog.apply(cv2.GaussianBlur(small, (BLUR, BLUR), 0))
        if n >= WARMUP:
            dets, mask, rej = detect(fg, road)
            for k in rej: rej_total[k] += rej[k]
        else:
            dets, mask = [], np.zeros((H, W), np.uint8)
        t_det = time.perf_counter()
        cars = tracker.update(dets, n, fps)
        for t in cars:
            all_tracks[t["id"]] = t
        counts = {d: sum(1 for c in tracker.crossings if c[3] == d) for d in DIRS}
        fps_now = len(times) / sum(times) if times else 0.0
        vis = small.copy(); draw(vis, cars, roi, counts, fps_now)
        out.write(np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        if show:
            cv2.imshow("Variant 1 - cars", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27): break
            if k == ord("p"): cv2.waitKey(0)
        dt = time.perf_counter() - t0; times.append(dt)
        areas = [t["area"] for t in cars]
        frames_rows.append([n, round(START_S + n / fps, 3), len(dets), len(cars), counts[DIRS[0]], counts[DIRS[1]],
                            round(np.mean(areas), 1) if areas else "", max(areas, default=""), min(areas, default=""),
                            round((t_det - t0) * 1000, 2), round(dt * 1000, 2), round(fps_now, 1)])
        for t in cars:
            obj_rows.append([n, t["id"], t["x"], t["y"], t["w"], t["h"], int(t["area"]), round(t["speed"], 1)])
        ok, frame = cap.read(); n += 1
    cap.release(); out.release()
    if show:
        cv2.destroyAllWindows()
    total_s = time.perf_counter() - t_loop

    def write(name, header, rows):
        with open(RESULTS / f"{name}{tag}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
    write("var1_frames", ["frame", "time_s", "candidates", "cars", f"count_{DIRS[0]}", f"count_{DIRS[1]}", "area_mean", "area_max",
                          "area_min", "detect_ms", "frame_ms", "fps"], frames_rows)
    write("var1_objects", ["frame", "id", "x", "y", "w", "h", "area", "speed_px_s"], obj_rows)
    write("var1_crossings", ["id", "frame", "time_s", "direction"], tracker.crossings)
    track_rows = [[i, round(START_S + t["first"] / fps, 2), round(START_S + t["last"] / fps, 2), t["hits"], round(t["path"], 1),
                   round(np.mean(t["areas"]), 1), t["counted"] or ""] for i, t in sorted(all_tracks.items())]
    write("var1_tracks", ["id", "first_s", "last_s", "frames_matched", "path_px", "mean_area", "crossed"], track_rows)
    stats = dict(video=VIDEO.name, frames=n, duration_s=round(n / fps, 1), ids_total=len(all_tracks),
                 counted_total=len(tracker.crossings), counted_per_direction=counts, start_s=START_S,
                 max_cars_in_frame=max((r[3] for r in frames_rows), default=0),
                 mean_cars_in_frame=round(float(np.mean([r[3] for r in frames_rows])), 2) if frames_rows else 0,
                 rejected=rej_total, mean_detect_ms=round(float(np.mean([r[9] for r in frames_rows])), 2),
                 mean_frame_ms=round(float(np.mean([r[10] for r in frames_rows])), 2),
                 system_fps=round(n / total_s, 1), window=show,
                 params=dict(HISTORY=HISTORY, VAR_THR=VAR_THR, MIN_AREA=MIN_AREA, MAX_AREA=MAX_AREA,
                             ASPECT=[ASPECT_MIN, ASPECT_MAX], MIN_SOLIDITY=MIN_SOLIDITY,
                             MAX_DIST=MAX_DIST, MIN_HITS=MIN_HITS, MAX_MISSED=MAX_MISSED))
    (RESULTS / f"var1_statistics{tag}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    if "--setup" in sys.argv:
        setup()
    else:
        main(show="--no-show" not in sys.argv)
