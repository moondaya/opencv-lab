"""Task 11: video -> preprocessing -> motion detection (MOG2) -> contours -> filtering -> tracking -> visualisation.
Usage (venv active, project folder):
    python task11_system.py             live window (p = pause, q = quit)
    python task11_system.py --no-show   no window
"""
import sys, csv, json, time, math
from pathlib import Path
from collections import deque
import cv2, numpy as np

VIDEO   = Path("data") / "task10_motion.mp4"
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

# ---- settings ----
MAX_SIDE, BLUR = 960, 5
HISTORY, VAR_THR = 100, 16
K_OPEN, K_CLOSE = 5, 15
MIN_AREA, MAX_AREA, MIN_SOLIDITY = 80, 20000, 0.7
ASPECT_MIN, ASPECT_MAX = 0.4, 2.5
MAX_DIST, MAX_MISSED, MIN_HITS, TRAIL = 60, 15, 5, 30
K_O = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_OPEN, K_OPEN))
K_C = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_CLOSE, K_CLOSE))
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---- 1) preprocessing, 2) motion mask, 3) contours, 4) filtering ----
def preprocess(frame):
    h, w = frame.shape[:2]; s = MAX_SIDE / max(h, w)
    small = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return small, cv2.GaussianBlur(small, (BLUR, BLUR), 0)

def clean(raw):
    m = np.where(raw == 255, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, K_O), cv2.MORPH_CLOSE, K_C)

def find_objects(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not MIN_AREA <= area <= MAX_AREA: continue
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or area / hull < MIN_SOLIDITY: continue
        if not ASPECT_MIN <= w / h <= ASPECT_MAX: continue
        out.append(dict(x=x, y=y, w=w, h=h, area=area, cx=x + w / 2, cy=y + h / 2))
    return out

# ---- 5) tracking ----
class CentroidTracker:
    def __init__(self, max_dist, max_missed, min_hits):
        self.max_dist, self.max_missed, self.min_hits = max_dist, max_missed, min_hits
        self.tracks, self.next_id = [], 1

    def update(self, dets):
        pairs = sorted((math.hypot(t["cx"] - d["cx"], t["cy"] - d["cy"]), ti, di)
                       for ti, t in enumerate(self.tracks) for di, d in enumerate(dets))
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:                                   # greedy: closest pairs first
            if dist > self.max_dist: break
            if ti in used_t or di in used_d: continue
            used_t.add(ti); used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            t.update({k: d[k] for k in ("x", "y", "w", "h", "area", "cx", "cy")})
            t["hits"] += 1; t["missed"] = 0; t["trail"].append((int(d["cx"]), int(d["cy"])))
            if t["id"] is None and t["hits"] >= self.min_hits:       # confirm -> give a visible ID
                t["id"] = self.next_id; self.next_id += 1
        for ti, t in enumerate(self.tracks):
            if ti not in used_t: t["missed"] += 1
        self.tracks = [t for t in self.tracks                        # tentative: 1 miss, confirmed: MAX_MISSED
                       if t["missed"] <= (self.max_missed if t["id"] is not None else 1)]
        for di, d in enumerate(dets):
            if di not in used_d:
                t = {k: d[k] for k in ("x", "y", "w", "h", "area", "cx", "cy")}
                t.update(id=None, hits=1, missed=0, trail=deque([(int(d["cx"]), int(d["cy"]))], maxlen=TRAIL))
                self.tracks.append(t)
        return [t for t in self.tracks if t["id"] is not None and t["missed"] == 0]

# ---- 6) visualisation ----
def id_colour(i):
    hsv = np.uint8([[[(i * 47) % 180, 230, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])

def text(img, s, org, col, scale=0.4):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)   # dark outline for readability
    cv2.putText(img, s, org, FONT, scale, col, 1, cv2.LINE_AA)

def draw_object(img, o):
    col = id_colour(o["id"]); x, y, w, h = o["x"], o["y"], o["w"], o["h"]
    cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
    if len(o["trail"]) > 1: cv2.polylines(img, [np.array(o["trail"], np.int32)], False, col, 2)
    lines = [f"Object ID: {o['id']}", f"X: {x}", f"Y: {y}", f"Width: {w}", f"Height: {h}", f"Area: {int(o['area'])}"]
    tx = x + w + 5 if x + w + 115 < img.shape[1] else max(x - 115, 0)
    for k, s in enumerate(lines): text(img, s, (tx, max(y, 10) + 12 + 13 * k), col)

def draw_panel(img, objs, fps):
    a = [o["area"] for o in objs]
    lines = [f"Objects in frame: {len(objs)}",
             f"Mean area: {np.mean(a):.0f}" if a else "Mean area: -",
             f"Max area: {max(a):.0f}" if a else "Max area: -",
             f"Min area: {min(a):.0f}" if a else "Min area: -",
             f"FPS: {fps:.1f}"]
    over = img.copy(); cv2.rectangle(over, (0, 0), (215, 20 * len(lines) + 8), (0, 0, 0), -1)
    cv2.addWeighted(over, 0.55, img, 0.45, 0, img)
    for k, s in enumerate(lines): text(img, s, (8, 20 + 20 * k), (255, 255, 255), 0.55)

# ---- main loop ----
def main(show=True):
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened(): sys.exit(f"Cannot open {VIDEO}")
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
    sub = cv2.createBackgroundSubtractorMOG2(history=HISTORY, varThreshold=VAR_THR, detectShadows=True)
    tracker = CentroidTracker(MAX_DIST, MAX_MISSED, MIN_HITS)
    writer, frame_rows, obj_rows, ms_hist, idx = None, [], [], deque(maxlen=30), 0
    if show: cv2.namedWindow("Task 11", cv2.WINDOW_NORMAL)

    while True:
        t0 = time.perf_counter()
        ok, f = cap.read()
        if not ok: break
        small, blur_c = preprocess(f)                       # 1 preprocessing
        mask = clean(sub.apply(blur_c))                     # 2 motion detection + morphology
        dets = find_objects(mask)                           # 3 contours + 4 filtering
        objs = tracker.update(dets)                         # 5 tracking
        fps_now = 1000 / np.mean(ms_hist) if ms_hist else 0.0
        vis = small.copy()                                  # 6 visualisation
        for o in objs: draw_object(vis, o)
        draw_panel(vis, objs, fps_now)
        if writer is None:
            writer = cv2.VideoWriter(str(RESULTS / "task11_output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_v, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        if show:
            cv2.imshow("Task 11", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                while (cv2.waitKey(30) & 0xFF) != ord("p"): pass
            elif key in (ord("q"), 27): break
        ms = (time.perf_counter() - t0) * 1000; ms_hist.append(ms)

        a = [o["area"] for o in objs]; ts = round(idx / fps_v, 3)
        frame_rows.append([idx, ts, len(objs), round(float(np.mean(a)), 1) if a else "", int(max(a)) if a else "",
                           int(min(a)) if a else "", round(ms, 2), round(fps_now, 1)])
        for o in objs:
            obj_rows.append([idx, ts, o["id"], o["x"], o["y"], o["w"], o["h"], int(o["area"]), int(o["cx"]), int(o["cy"])])
        idx += 1

    cap.release(); writer.release(); cv2.destroyAllWindows()
    with open(RESULTS / "task11_frames.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "time_s", "n_objects", "mean_area", "max_area", "min_area", "frame_ms", "fps"])
        w.writerows(frame_rows)
    with open(RESULTS / "task11_objects.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "time_s", "id", "x", "y", "w", "h", "area", "cx", "cy"]); w.writerows(obj_rows)

    tracks = {}
    for r in obj_rows: tracks.setdefault(r[2], []).append(r)
    with open(RESULTS / "task11_tracks.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "first_s", "last_s", "frames_seen", "mean_area"])
        for i, rs in sorted(tracks.items()):
            w.writerow([i, rs[0][1], rs[-1][1], len(rs), round(float(np.mean([r[7] for r in rs])), 1)])
    ms_all = np.array([r[6] for r in frame_rows]); n = np.array([r[2] for r in frame_rows])
    summary = dict(frames=len(frame_rows), ids_created=tracker.next_id - 1, max_objects=int(n.max()),
                   mean_objects=round(float(n.mean()), 2), mean_frame_ms=round(float(ms_all.mean()), 1),
                   system_fps=round(1000 / float(ms_all.mean()), 1))
    (RESULTS / "task11_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    for i, rs in sorted(tracks.items()):
        print(f"  ID {i:2d}: {rs[0][1]:5.1f}-{rs[-1][1]:5.1f} s, {len(rs):4d} frames, mean area {np.mean([r[7] for r in rs]):6.0f}")

if __name__ == "__main__":
    main(show="--no-show" not in sys.argv)