import cv2
print("OpenCV", cv2.__version__)

# 1) which tracker classes exist in this build
print("cv2:", sorted(n for n in dir(cv2) if "Tracker" in n))
if hasattr(cv2, "legacy"):
    print("cv2.legacy:", sorted(n for n in dir(cv2.legacy) if "Tracker" in n))

# 2) can each one be created?
for name in ["TrackerCSRT", "TrackerKCF", "TrackerMIL"]:
    for label, factory in [(f"cv2.{name}.create", getattr(getattr(cv2, name, None), "create", None)),
                           (f"cv2.{name}_create", getattr(cv2, name + "_create", None))]:
        if factory is None:
            print(f"{label:28s} not available")
            continue
        try:
            factory(); print(f"{label:28s} OK")
        except Exception as e:
            print(f"{label:28s} ERROR: {e}")

# 3) video backends (reading/writing mp4 needs FFMPEG or MSMF)
info = cv2.getBuildInformation()
print([line.strip() for line in info.splitlines() if "FFMPEG" in line or "MSMF" in line])