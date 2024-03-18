import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./_frame_75.jpg", 0)
plt.imshow(img)
plt.show()
# kernel = np.zeros((5, 5), np.float32)

# gaussian_blur = cv2.GaussianBlur(img, (3, 3), 0)
# median_blur = cv2.medianBlur(img, 3)

# cv2.imshow("orig", img)
# cv2.imshow("filter", gaussian_blur)
# cv2.imshow("filter2", median_blur)

edges = cv2.Canny(img, 100, 200)

cv2.imshow("edges", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
