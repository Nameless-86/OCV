import cv2
import numpy as np
import matplotlib.pyplot as plt


# Definición de la función para la ecualización local de histograma
def ecualizacion_local_histograma(imagen, tamano_ventana):
    alto, ancho = imagen.shape
    mitad_ventana = tamano_ventana // 2

    # Agregar bordes a la imagen para manejar los píxeles cerca de los bordes
    imagen_con_bordes = cv2.copyMakeBorder(
        imagen,
        mitad_ventana,
        mitad_ventana,
        mitad_ventana,
        mitad_ventana,
        cv2.BORDER_REPLICATE,
    )
    # print(imagen_con_bordes)

    # Matriz vacía para almacenar los resultados
    imagen_resultado = np.empty(imagen.shape)

    # Recorremos la imagen original
    for i in range(mitad_ventana, alto + mitad_ventana):
        for j in range(mitad_ventana, ancho + mitad_ventana):
            # Definimos una ventana deslizante en la imagen con el tamaño especificado
            ventana = imagen_con_bordes[
                i - mitad_ventana : i + mitad_ventana + 1,
                j - mitad_ventana : j + mitad_ventana + 1,
            ]
            # Aplicamos la ecualización del histograma a la ventana

            ventana_equ = cv2.equalizeHist(ventana)

            # Almacenamos el valor ecualizado en la posición correspondiente de la imagen de resultado
            imagen_resultado[i - mitad_ventana, j - mitad_ventana] = ventana_equ[
                mitad_ventana, mitad_ventana
            ]

    #  print(imagen_resultado)

    return imagen_resultado


# Ruta de la imagen de entrada
ruta_imagen = "imagen.tif"

# Lectura de la imagen en escala de grises
imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
imagen_original.shape


# Aplicamos la ecualización local de histograma a tres tamaños de ventana diferentes
img1 = ecualizacion_local_histograma(imagen_original, 3 * 3)

img2 = ecualizacion_local_histograma(imagen_original, 11 * 3)

img3 = ecualizacion_local_histograma(imagen_original, 33 * 3)


# Configuración de subplots para mostrar las imágenes
ax1 = plt.subplot(221)
plt.title("Imagen Original")
plt.imshow(imagen_original)
plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title("Ventana de 3x3")
plt.imshow(img1, cmap="gray")
plt.subplot(223, sharex=ax1, sharey=ax1)
plt.title("Ventana de 9x3")
plt.imshow(img2, cmap="gray")
plt.subplot(224, sharex=ax1, sharey=ax1)
plt.title("Ventana de 33x3")
plt.imshow(img3, cmap="gray")

# Mostrar el resultado
plt.show()

#############AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
hist = cv2.calcHist([imagen_original], [0], None, [256], [0, 256])

hist1, bins = np.histogram(img1.ravel(), bins=256, range=[0, 256])

hist2, bins = np.histogram(img2.ravel(), bins=256, range=[0, 256])

hist3, bins = np.histogram(img3.ravel(), bins=256, range=[0, 256])


ax1 = plt.subplot(221)
plt.title("Imagen Original")
plt.xlim(0, 256)
plt.grid()
plt.plot(hist)
plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title("Ventana de 3*3")
plt.xlim(0, 256)
plt.plot(hist1)
plt.grid()
plt.subplot(223, sharex=ax1, sharey=ax1)
plt.title("Ventana de 11*3")
plt.xlim(0, 256)
plt.plot(hist2)
plt.grid()
plt.subplot(224, sharex=ax1, sharey=ax1)
plt.title("Ventana de 33*3")
plt.xlim(0, 256)
plt.plot(hist3)
plt.grid()
plt.show()
