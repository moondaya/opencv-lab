"""Task 9: single-object tracking with CSRT / KCF / MIL (OpenCV 5.0)

Usage (venv active, in the project folder):
    python task09_tracking.py CSRT              first run: select object, saved to results/task09_roi.json
    python task09_tracking.py KCF               reuses the saved start frame + box
    python task09_tracking.py MIL               optional third tracker
    python task09_tracking.py CSRT --reselect   choose the object again
Keys: s = select (selection mode), p = pause/resume, q / Esc = quit
"""
import sys, json, csv, time
from pathlib import Path
import cv2

VIDEO    = Path("data") / "task09_track.mp4"
RESULTS  = Path("results"); RESULTS.mkdir(exist_ok=True)
ROI_FILE = RESULTS / "task09_roi.json"
MAX_SIDE = 960                       # longer side of processed frames, px (portrait 1080x1920 -> 540x960)
WIN      = "Task 9 tracking"

TRACKERS = {"CSRT": cv2.TrackerCSRT.create,
            "KCF":  cv2.TrackerKCF.create,
            "MIL":  cv2.TrackerMIL.create}

def resize(frame):
    h, w = frame.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame

def open_video():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"Cannot open {VIDEO}")
    return cap

def choose_roi():
    """Play the video; press s when the object is visible, then drag a box and press Enter."""
    cap, idx = open_video(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("Video ended before an object was selected")
        frame = resize(frame)
        show = frame.copy()
        cv2.putText(show, f"frame {idx}: s = select, q = quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(WIN, show)
        key = cv2.waitKey(33) & 0xFF
        if key == ord("s"):
            roi = cv2.selectROI(WIN, frame, showCrosshair=True, fromCenter=False)
            if roi[2] > 0 and roi[3] > 0:                       # (0,0,0,0) = cancelled
                cap.release()
                data = {"start_frame": idx, "roi": [int(v) for v in roi], "max_side": MAX_SIDE}
                ROI_FILE.write_text(json.dumps(data, indent=2))
                print("saved", data, "->", ROI_FILE)
                return data
        elif key in (ord("q"), 27):
            sys.exit("cancelled")
        idx += 1

def main():
    name = (sys.argv[1] if len(sys.argv) > 1 else "CSRT").upper()
    if name not in TRACKERS:
        sys.exit(f"Unknown tracker {name}; use one of {list(TRACKERS)}")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    if "--reselect" in sys.argv or not ROI_FILE.exists():
        sel = choose_roi()
    else:
        sel = json.loads(ROI_FILE.read_text())
        print("using saved selection", sel)
    start, roi = sel["start_frame"], tuple(sel["roi"])

    cap = open_video()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    for _ in range(start):                          # skip to the selection frame, frame by frame (exact)
        cap.read()
    ok, frame = cap.read()
    if not ok:
        sys.exit("Cannot read the start frame")
    frame = resize(frame)
    H, W = frame.shape[:2]

    tracker = TRACKERS[name]()
    tracker.init(frame, roi)
    out_path = RESULTS / f"task09_{name.lower()}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rows, idx, paused = [], start, False
    box, ok_trk, ms = roi, True, 0.0                # first frame shows the selected box
    while True:
        if not paused:
            if idx > start:
                ok, frame = cap.read()
                if not ok:
                    break                           # end of video
                frame = resize(frame)
                t0 = time.perf_counter()
                ok_trk, box = tracker.update(frame)
                ms = (time.perf_counter() - t0) * 1000
                box = tuple(int(v) for v in box)
            x, y, w, h = box
            cx, cy = x + w // 2, y + h // 2
            rows.append([idx, round(idx / fps, 3), int(ok_trk), x, y, w, h, cx, cy, round(ms, 2)])

            vis = frame.copy()
            if ok_trk:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(vis, f"X={x} Y={y} W={w} H={h}", (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(vis, "TRACKING LOST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            fps_now = 1000 / ms if ms > 0 else 0
            cv2.putText(vis, f"{name} | frame {idx} | {fps_now:5.1f} FPS", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            writer.write(vis)
            cv2.imshow(WIN, vis)
            idx += 1
        key = cv2.waitKey(30 if paused else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            paused = not paused

    cap.release(); writer.release(); cv2.destroyAllWindows()

    csv_path = RESULTS / f"task09_{name.lower()}.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "ok", "x", "y", "w", "h", "cx", "cy", "update_ms"])
        wr.writerows(rows)

    tracked = rows[1:]                              # exclude the init frame
    n = max(len(tracked), 1)
    n_ok = sum(r[2] for r in tracked)
    first_fail = next((r[0] for r in tracked if r[2] == 0), None)
    mean_ms = sum(r[9] for r in tracked) / n
    summary = dict(tracker=name, start_frame=start, frames=len(tracked),
                   success_pct=round(100 * n_ok / n, 1), first_failure_frame=first_fail,
                   mean_update_ms=round(mean_ms, 2), fps=round(1000 / mean_ms, 1) if mean_ms else None)
    (RESULTS / f"task09_{name.lower()}_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    print("saved:", out_path, "|", csv_path)

if __name__ == "__main__":
    main()