"""Task 10: motion -> mask -> morphology -> contours -> boxes -> count of moving objects.
Usage (venv active, project folder):
    python task10_motion.py mog2_100              one method, with a live window (q = quit, p = pause)
    python task10_motion.py all --no-show         all four methods, no window (faster)
Methods: a1 (absdiff vs empty frame), a2 (absdiff vs previous frame), mog2_500, mog2_100
"""
import sys, csv, json, time
from pathlib import Path
import cv2, numpy as np

VIDEO   = Path("data") / "task10_motion.mp4"
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
MAX_SIDE, BLUR, THR, REF_TIME = 960, 5, 25, 1.0
K_OPEN, K_CLOSE, MIN_AREA = 5, 15, 80
METHODS = ["a1", "a2", "mog2_500", "mog2_100"]
K_O = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_OPEN, K_OPEN))
K_C = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_CLOSE, K_CLOSE))

def prep(frame):
    h, w = frame.shape[:2]; s = MAX_SIDE / max(h, w)
    small = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    blur_c = cv2.GaussianBlur(small, (BLUR, BLUR), 0)
    return small, blur_c, cv2.cvtColor(blur_c, cv2.COLOR_BGR2GRAY)

def run(method, show=True):
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened(): sys.exit(f"Cannot open {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sub = (cv2.createBackgroundSubtractorMOG2(history=int(method.split("_")[1]), varThreshold=16,
                                              detectShadows=True) if method.startswith("mog2") else None)
    ref = prev = writer = None
    rows, idx, paused = [], 0, False
    if show: cv2.namedWindow(method, cv2.WINDOW_NORMAL)

    while True:
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("p"): paused = False
            elif key in (ord("q"), 27): break
            continue
        ok, f = cap.read()
        if not ok: break
        t0 = time.perf_counter()
        small, blur_c, gray = prep(f)

        # 1) motion mask
        if method == "a1":
            if idx == int(REF_TIME * fps): ref = gray                       # empty reference frame
            mask = (np.where(cv2.absdiff(gray, ref) > THR, 255, 0).astype(np.uint8)
                    if ref is not None else np.zeros_like(gray))
        elif method == "a2":
            mask = (np.where(cv2.absdiff(gray, prev) > THR, 255, 0).astype(np.uint8)
                    if prev is not None else np.zeros_like(gray))
        else:
            mask = np.where(sub.apply(blur_c) == 255, 255, 0).astype(np.uint8)   # drop shadows (127)
        prev = gray

        # 2) morphology  3) contours + area filter  4) boxes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_O)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_C)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) + (cv2.contourArea(c),) for c in cnts if cv2.contourArea(c) >= MIN_AREA]
        ms = (time.perf_counter() - t0) * 1000                               # processing time only

        # 5) visualisation + output video
        vis = small.copy()
        for i, (x, y, w, h, a) in enumerate(boxes, 1):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"{i}: {int(a)}px", (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(vis, f"{method} | Moving objects: {len(boxes)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"t={idx / fps:5.1f}s  {1000 / ms if ms else 0:5.0f} FPS", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        combo = np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        if writer is None:
            writer = cv2.VideoWriter(str(RESULTS / f"task10_{method}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (combo.shape[1], combo.shape[0]))
        writer.write(combo)
        rows.append([idx, round(idx / fps, 3), len(boxes), int(sum(b[4] for b in boxes)), round(ms, 2)])

        if show:
            cv2.imshow(method, combo)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break
            if key == ord("p"): paused = True
        idx += 1

    cap.release(); writer.release()
    if show: cv2.destroyWindow(method)

    with open(RESULTS / f"task10_{method}.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["frame", "time_s", "count", "total_area_px", "ms"]); wr.writerows(rows)
    counts = np.array([r[2] for r in rows]); ms_all = np.array([r[4] for r in rows])
    summary = dict(method=method, frames=len(rows), mean_ms=round(float(ms_all.mean()), 2),
                   fps=round(1000 / float(ms_all.mean()), 1), max_count=int(counts.max()),
                   mean_count=round(float(counts.mean()), 2), frames_with_motion_pct=round(100 * float((counts > 0).mean()), 1))
    (RESULTS / f"task10_{method}_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "mog2_100"
    show = "--no-show" not in sys.argv
    for m in (METHODS if arg == "all" else [arg]):
        if m not in METHODS: sys.exit(f"unknown method {m}; use {METHODS} or all")
        run(m, show)