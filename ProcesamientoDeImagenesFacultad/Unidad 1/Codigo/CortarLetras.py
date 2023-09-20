import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lectura

img = cv2.imread("letras.png", cv2.IMREAD_GRAYSCALE)

# Mostrar
plt.imshow(img, cmap="gray")
plt.show()

# Recorte

img_crop = img[20:100, 24:91]  # Y #X
plt.figure()
plt.imshow(img_crop, cmap="gray")
plt.show(block=False)

img_crop = img[20:100, 112:180]  # Y #X
plt.figure()
plt.imshow(img_crop, cmap="gray")
plt.show(block=False)


################################

################################

################################
# Filas y columnas
img_zeros = img == 0
plt.imshow(img_zeros, cmap="gray")
plt.show()

# columnas
img_col_zeros = img_zeros.any(axis=0)
img_col_zeros_idxs = np.argwhere(img_zeros.any(axis=0))
plt.imshow(img, cmap="gray")
xc = np.arange(img.shape[1])
yc = img_col_zeros * (img.shape[0] - 1)
plt.plot(xc, yc, c="b")
plt.show(block=False)

# Filas

img_row_zeros = img_zeros.any(axis=1)
img_row_zeros_idx = np.argwhere(img_zeros.any(axis=1))
plt.imshow(img, cmap="gray")
xr = np.arange(img.shape[0])
yr = img_row_zeros * (img.shape[1] - 1)
plt.plot(yr, xr, c="r")
plt.show(block=False)

##########################

################################

################################

################################
rows, cols = np.where(img == 0)  # Assuming black is represented by pixel value 0
top_row, left_col = min(rows), min(cols)
bottom_row, right_col = max(rows), max(cols)

# Crop the image using NumPy slicing
cropped_image = img[top_row : bottom_row + 1, left_col : right_col + 1]

cv2.imshow("Cropped Image", cropped_image)
