import matplotlib.pyplot as plt
import numpy as np
import cv2

patentes = [f'img/patentes/img{i:02}.png' for i in range(1, 13)]

def prueba(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    blur = cv2.GaussianBlur(img, (1, 21), 0)
    # plt.imshow(blur, cmap='gray'), plt.show()
    img_canny = cv2.Canny(blur, 250, 300)
    # plt.imshow(img_canny, cmap='gray'), plt.show()

    elemento_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 1))
    img_cierre = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, elemento_cierre)

    n, labels, stats, _  = cv2.connectedComponentsWithStats(img_cierre)

    filtro = (
        (stats[:, cv2.CC_STAT_AREA] >= 400) & 
        (stats[:, cv2.CC_STAT_HEIGHT] < stats[:, cv2.CC_STAT_WIDTH]))

    labels_filtrado = np.argwhere(filtro).flatten().tolist()

    idx_patente = labels_filtrado[-1]
    stats_patente = stats[idx_patente]


    coor_h = stats_patente[cv2.CC_STAT_LEFT] 
    coor_v = stats_patente[cv2.CC_STAT_TOP]

    ancho  = stats_patente[cv2.CC_STAT_WIDTH]   
    largo  = stats_patente[cv2.CC_STAT_HEIGHT]

    sub_imagen = \
            img[coor_v:coor_v + largo, coor_h: coor_h + ancho]


    mascara = (labels == idx_patente).astype('uint8') * 255

    elemento_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

    img_dil = cv2.dilate(mascara, elemento_dil)
    
    # ax = plt.subplot(221)
    # plt.imshow(img, cmap='gray')
    # plt.title(path)
    # plt.subplot(222, sharex=ax, sharey=ax)
    # plt.imshow(mascara, cmap='gray')
    # plt.subplot(223, sharex=ax, sharey=ax)
    # plt.imshow(img_dil, cmap='gray')
    # plt.subplot(224, sharex=ax, sharey=ax)
    # plt.imshow(np.bitwise_and(img, img_dil), cmap='gray')
    # plt.show()

    return np.bitwise_and(img, img_dil), sub_imagen

crops = list()

for patente in patentes:
    crops.append(prueba(patente))

def contar_letras(img):
    '''
    Recibe una imagen umbralada y cuenta cuantas letras tiene.
    '''

    n, _, stats, _ = cv2.connectedComponentsWithStats(img)
    
    letras = 0
    for i in range(1, n):
        if (stats[i][-1] < 100 and stats[i][-1] > 15) and (stats[i][2] < stats[i][3]):
            letras += 1

    return letras

def umbralar(sub_imagen, imagen):
    mediana = np.median(sub_imagen) * 0.50
    suma = 0

    encontrado = False

    while not encontrado:
        umbralada = (imagen > (mediana + suma)).astype('uint8')
        cantidad_letras = contar_letras(umbralada)

        if cantidad_letras != 6:
            suma += 1
        else:
            encontrado = True
    
    return mediana + suma

def graficar_patente(sub_img, img):

    umbral = umbralar(sub_img, img)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((img > umbral).astype('uint8'))

    img_vacia = np.zeros_like(img > umbral, dtype='uint8')

    for i in range(1, n):

        if (stats[i][-1] < 100 and stats[i][-1] > 15) and (stats[i][2] < stats[i][3]):
            img_vacia[labels == i] = 1

    plt.imshow(img_vacia, cmap='gray'), plt.show()

for crop in crops:
    graficar_patente(crop[1], crop[0])
    