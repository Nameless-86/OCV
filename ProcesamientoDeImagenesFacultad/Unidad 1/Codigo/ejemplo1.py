import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img1 = cv.imread("xray-chest.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("cameraman.tif", cv.IMREAD_GRAYSCALE)

# --- Matplotlib -------------------------------------------------
plt.figure(1)
h = plt.imshow(img1, cmap="gray")
plt.colorbar(h)

plt.figure(2)
h = plt.imshow(img2, cmap="gray", vmin=0, vmax=255)
plt.colorbar(h)

plt.show()
# plt.show(block=False)

# --- Open CV -----------------------------------------------------
cv.imshow("Imagen 1", img1)
cv.imshow("Imagen 2", img2)
cv.waitKey(0)
cv.destroyAllWindows()
