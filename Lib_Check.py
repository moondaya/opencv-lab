import sys, os
from importlib import metadata

print("Python:", sys.version.split()[0], "|", sys.executable)

# Core libraries
for name in ["cv2", "numpy", "matplotlib"]:
    try:
        mod = __import__(name)
        print(f"[OK] {name} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"[MISSING] {name}")

# Which OpenCV packages are installed (only ONE should be present)
pkgs = [d.metadata["Name"] for d in metadata.distributions()
        if d.metadata["Name"] and d.metadata["Name"].lower().startswith("opencv")]
print("OpenCV packages:", pkgs or "none")
if len(pkgs) > 1:
    print("[WARNING] Several OpenCV packages installed - keep only opencv-contrib-python")

try:
    import cv2
    # Trackers (Task 9): CSRT/KCF need the contrib package
    for t in ["TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"]:
        found = hasattr(cv2, t) or hasattr(getattr(cv2, "legacy", None), t)
        print(f"[{'OK' if found else 'MISSING'}] {t}")

    # Haar cascade file (Task 8)
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    print(f"[{'OK' if os.path.exists(path) else 'MISSING'}] Haar cascade file")

    # GUI support for imshow / trackbars / selectROI (Tasks 5, 9)
    gui = [l.strip() for l in cv2.getBuildInformation().splitlines() if "GUI:" in l]
    print("GUI backend:", gui[0] if gui else "unknown")
except ImportError:
    pass