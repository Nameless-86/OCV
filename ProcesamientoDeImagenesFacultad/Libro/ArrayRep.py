from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = np.array(Image.open("foton.jpg"))
print(im.shape, im.dtype)  # (954, 960, 3) uint8 (row, col, color channel) data type
## Acceso ##
i, j, k = 3
value = im[i, j, k]
# set the values of row i with values from row j
im[i, :] = im[j, :]
# set all values in column i to 100
im[:, i] = 100
# the sum of the values of the first 100 rows and 50 columns
im[:100, :50].sum()
# rows 50-100, columns 50-100 (100th not included)
im[50:100, 50:100]
# average of row i
im[i].mean()
# last column
im[:, -1]
# second to last row
im[-2, :](im[-2])

im2 = np.array(Image.open("foton.jpg").convert("L"), "f")  # (954, 960) float32
print(im2.shape, im2.dtype)
