import cv2
import numpy as np

img = np.zeros((200, 400, 3), np.uint8)
cv2.putText(img, "GUI works - press any key", (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.namedWindow("GUI test")
cv2.createTrackbar("Test", "GUI test", 50, 100, lambda v: None)  # needed for Task 5
cv2.imshow("GUI test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()