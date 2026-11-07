"""Task 12, step 1: mark the 9 objects by hand (ground truth).
For each object: drag a box tightly around it, then press Enter (or Space).
When all objects are marked, press Esc.
"""
import json
from pathlib import Path
import cv2, numpy as np

IMG = Path("data") / "task03_objects.jpg"
OUT = Path("results") / "task12_gt_boxes.json"; OUT.parent.mkdir(exist_ok=True)
W = 800                                                   # same size as Task 3

img = cv2.imread(str(IMG)); assert img is not None, f"Cannot read {IMG}"
h, w = img.shape[:2]
img = cv2.resize(img, (W, round(h * W / w)), interpolation=cv2.INTER_AREA)

win = "Mark objects: drag + Enter for each, Esc when done"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
boxes = np.array(cv2.selectROIs(win, img, showCrosshair=False)).reshape(-1, 4)
cv2.destroyAllWindows()

boxes = [[int(v) for v in b] for b in boxes if b[2] > 0 and b[3] > 0]
OUT.write_text(json.dumps({"image_width": W, "boxes": boxes}, indent=2))

vis = img.copy()
for i, (x, y, bw, bh) in enumerate(boxes, 1):
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
    cv2.putText(vis, str(i), (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.imwrite(str(Path("results") / "task12_gt.png"), vis)
print(f"{len(boxes)} boxes saved -> {OUT}")