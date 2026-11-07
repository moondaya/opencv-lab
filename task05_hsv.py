# task05_hsv.py — Task 5: colour segmentation with interactive HSV trackbars
from pathlib import Path
import json
import cv2
import numpy as np

IMAGE_PATH = Path("data") / "task03_objects.jpg"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TARGET_WIDTH = 800
MIN_AREA = 300
WIN = "HSV controls"

# --- load, resize, convert ---
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f"Could not read the image: {IMAGE_PATH}")
h, w = image.shape[:2]
img = cv2.resize(image, (TARGET_WIDTH, round(h * TARGET_WIDTH / w)), interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)   # 1. BGR -> HSV

# --- trackbars: 6 limits, start values = medium range for the marker ---
def nothing(_):
    pass

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 500, 300)
for name, start, maxval in [("H min", 75, 179), ("H max", 100, 179),
                            ("S min", 80, 255), ("S max", 255, 255),
                            ("V min", 30, 255), ("V max", 255, 255)]:
    cv2.createTrackbar(name, WIN, start, maxval, nothing)

print("Move the sliders. Keys: s = save results, q or Esc = quit")

# --- replace the while-loop of Step 1 with this ---
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
invert = False          # i = toggle: False -> object by colour; True -> everything EXCEPT the range
save_count = 0

print("Keys: s = save, i = invert mask (background mode), q / Esc = quit")

while True:
    hmin, hmax = cv2.getTrackbarPos("H min", WIN), cv2.getTrackbarPos("H max", WIN)
    smin, smax = cv2.getTrackbarPos("S min", WIN), cv2.getTrackbarPos("S max", WIN)
    vmin, vmax = cv2.getTrackbarPos("V min", WIN), cv2.getTrackbarPos("V max", WIN)
    lower, upper = np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax])

    mask = cv2.inRange(hsv, lower, upper)                                   # 3. mask
    if invert:
        mask = cv2.bitwise_not(mask)                                        # background mode
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)                   # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)                  # fill small gaps

    extracted = cv2.bitwise_and(img, img, mask=mask)                        # 4. extract
    result = img.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 5. contours
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    info = f"H {hmin}-{hmax}  S {smin}-{smax}  V {vmin}-{vmax}  {'INVERTED' if invert else ''}"

    if contours:
        if invert:
            targets = contours                                              # all objects
        else:
            targets = [max(contours, key=cv2.contourArea)]                  # the largest = target
        for c in targets:
            x, y, bw, bh = cv2.boundingRect(c)                              # 6. rectangle
            cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.drawContours(result, [c], -1, (0, 0, 255), 1)
        main = targets[0] if not invert else max(targets, key=cv2.contourArea)
        c_area = cv2.contourArea(main)                                      # 7. area (contour)
        single = np.zeros_like(mask)
        cv2.drawContours(single, [main], -1, 255, -1)
        px_area = cv2.countNonZero(cv2.bitwise_and(single, mask))           # 7. area (mask pixels)
        x, y, bw, bh = cv2.boundingRect(main)
        info2 = (f"regions >= {MIN_AREA}px: {len(contours)} | largest: contour area {c_area:.0f} px2, "
                 f"mask pixels {px_area}, box {bw}x{bh} at ({x},{y})")
    else:
        info2 = "no region >= MIN_AREA"

    cv2.putText(result, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(result, info2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    cv2.imshow("Mask", mask)
    cv2.imshow("Extracted", extracted)
    cv2.imshow("Result", result)

    key = cv2.waitKey(30) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("i"):
        invert = not invert
    if key == ord("s"):
        save_count += 1
        tag = f"task05_{'background' if invert else 'object'}_{save_count}"
        cv2.imwrite(str(RESULTS_DIR / f"{tag}_mask.png"), mask)
        cv2.imwrite(str(RESULTS_DIR / f"{tag}_extracted.png"), extracted)
        cv2.imwrite(str(RESULTS_DIR / f"{tag}_result.png"), result)
        params = {"H": [hmin, hmax], "S": [smin, smax], "V": [vmin, vmax], "inverted": invert,
                  "regions": len(contours), "info": info2}
        (RESULTS_DIR / f"{tag}_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        print(f"Saved {tag}: {info} | {info2}")

cv2.destroyAllWindows()

