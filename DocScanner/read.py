import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./cap.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img)

plt.show()
