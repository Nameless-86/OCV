import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lectura

img = cv2.imread("thisone.jpg", cv2.IMREAD_GRAYSCALE)

# # informacion
type(img)
print(img.dtype)
print(img.shape)
w, h = img.shape


# stats
img.min()
img.max()
pix_vals = np.unique(img)
N_pix_vals = len(np.unique(img))

# Mostrar
plt.imshow(img, cmap="gray")
plt.show()

h = plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.colorbar(h)
plt.title("Imagen")
plt.xlabel("X")
plt.ylabel("Y")
plt.xticks([])
plt.yticks([])
plt.show()

h = plt.imshow(img, cmap="gray", vmin=0, vmax=100)
plt.colorbar(h)
plt.show()

plt.subplot(121)
h = plt.imshow(img, cmap="gray")
plt.colorbar(h)
plt.title("Normalizada")
plt.subplot(122)
h = plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.colorbar(h)
plt.title("Sin normalizar")
plt.show()

# Generando imagenes variando la calidad
