"""Variant 1: ground truth - press a key each time a car crosses the counting line.

Keys: 1 = car crossed in the first direction (L->R for a steep line, T->B otherwise),
      2 = second direction (R->L or B->T), u = undo last, space = pause,
      , / . = slower / faster, q = quit and save.
Output: results/var1_gt_crossings.csv (frame, time_s, direction); time_s = time in the original video.
"""
from pathlib import Path
import json, csv
import cv2

VIDEO = Path("data") / "var1_cars.mp4"
RESULTS = Path("results")
START_S = 28.0                  # same value as in var1_cars_system.py
MAX_SIDE = 960

roi = json.loads((RESULTS / "var1_roi.json").read_text())
(ax, ay), (bx, by) = roi["line"]
DIRS = ("L->R", "R->L") if abs(bx - ax) < abs(by - ay) else ("T->B", "B->T")

cap = cv2.VideoCapture(str(VIDEO)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
for _ in range(int(round(START_S * fps))):
    cap.grab()
marks, n, delay, paused = [], 0, int(1000 / fps), False
ok, frame = cap.read()
while ok:
    h, w = frame.shape[:2]; s = MAX_SIDE / max(h, w)
    vis = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.line(vis, (ax, ay), (bx, by), (0, 0, 255), 2)
    c1 = sum(m[2] == DIRS[0] for m in marks); c2 = len(marks) - c1
    cv2.putText(vis, f"t={START_S + n / fps:5.1f}s  [1] {DIRS[0]}: {c1}  [2] {DIRS[1]}: {c2}  delay {delay} ms",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("ground truth", vis)
    k = cv2.waitKey(0 if paused else delay) & 0xFF
    if k == ord("1"): marks.append((n, round(START_S + n / fps, 3), DIRS[0]))
    elif k == ord("2"): marks.append((n, round(START_S + n / fps, 3), DIRS[1]))
    elif k == ord("u") and marks: marks.pop()
    elif k == ord(" "): paused = not paused
    elif k == ord(","): delay = min(delay * 2, 400)
    elif k == ord("."): delay = max(delay // 2, 5)
    elif k in (ord("q"), 27): break
    if not paused:
        ok, frame = cap.read(); n += 1
cap.release(); cv2.destroyAllWindows()
with open(RESULTS / "var1_gt_crossings.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["frame", "time_s", "direction"]); wr.writerows(marks)
print(f"saved {len(marks)} crossings: {DIRS[0]} {sum(m[2] == DIRS[0] for m in marks)}, "
      f"{DIRS[1]} {sum(m[2] == DIRS[1] for m in marks)}")
