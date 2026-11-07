"""Label the number of objects per frame while the video plays.
Usage:  python task10_annotate.py moving     (count only objects that are moving now)
        python task10_annotate.py visible    (count all objects in view, moving or resting)
Keys:   0-9 = set current count (kept until you press another number)
        space = pause/resume   q / Esc = save and quit
"""
import sys, csv
from pathlib import Path
import cv2

VIDEO = Path("data") / "task10_motion.mp4"
KIND = sys.argv[1] if len(sys.argv) > 1 else "moving"
OUT = Path("results") / f"task10_gt_{KIND}.csv"; OUT.parent.mkdir(exist_ok=True)
DELAY_MS = 66                                   # ~half speed for a 30 fps video

cap = cv2.VideoCapture(str(VIDEO)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
cv2.namedWindow(KIND, cv2.WINDOW_NORMAL)
labels, count, paused, frame, idx = [], 0, False, None, 0
while True:
    if not paused:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]; s = 960 / max(h, w)
        frame = cv2.resize(frame, (round(w * s), round(h * s)))
        labels.append((idx, round(idx / fps, 3), count)); idx += 1
    show = frame.copy()
    cv2.putText(show, f"{KIND}: {count}   t={idx / fps:5.1f}s   {'PAUSED' if paused else ''}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow(KIND, show)
    key = cv2.waitKey(30 if paused else DELAY_MS) & 0xFF
    if ord("0") <= key <= ord("9"):
        count = key - ord("0")
        if labels: labels[-1] = (labels[-1][0], labels[-1][1], count)   # apply to current frame too
    elif key == ord(" "): paused = not paused
    elif key in (ord("q"), 27): break
cap.release(); cv2.destroyAllWindows()

with open(OUT, "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["frame", "time_s", KIND]); wr.writerows(labels)
print(f"saved {len(labels)} labels -> {OUT}")