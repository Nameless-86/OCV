import matplotlib.pyplot as plt
import numpy as np
import cv2

# Carga de imagen
PATH_MONEDAS = "./img/monedas.jpg"

img = cv2.imread(PATH_MONEDAS)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Funciones auxiliares
def contar_circulos(imagen):
    imagen = cv2.medianBlur(imagen, 7)

    circles = cv2.HoughCircles(
        imagen,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=50,
        param2=50,
        minRadius=20,
        maxRadius=50,
    )

    n = 0

    if isinstance(circles, np.ndarray):
        n = len(circles[0])

    return n


def graficar_caja(img, stats, color):
    cv2.rectangle(
        img,
        (stats[0], stats[1]),
        (stats[0] + stats[2], stats[1] + stats[3]),
        color=color,
        thickness=3,
    )


blur = cv2.GaussianBlur(img_gray, (3, 3), 2)
img_canny = cv2.Canny(blur, 50, 200)
plt.imshow(img_canny, cmap="gray"), plt.show()


def imfillhole_v2(img):
    img_flood_fill = img.copy().astype("uint8")  # Genero la imagen de salida
    h, w = img.shape[:2]  # Genero una mascara necesaria para cv2.floodFill()
    mask = np.zeros(
        (h + 2, w + 2), np.uint8
    )  # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    cv2.floodFill(img_flood_fill, mask, (0, 0), 255)  # Relleno o inundo la imagen.
    img_flood_fill_inv = cv2.bitwise_not(
        img_flood_fill
    )  # Tomo el complemento de la imagen inundada --> Obtenog SOLO los huecos rellenos.
    img_fh = (
        img | img_flood_fill_inv
    )  # La salida es un OR entre la imagen original y los huecos rellenos.
    return img_fh


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 45))
img2 = cv2.dilate(img_canny, kernel)
img2 = imfillhole_v2(img2)

plt.imshow(img2, cmap="gray"), plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

img2 = cv2.erode(img2, kernel)
plt.imshow(img2, cmap="gray"), plt.show(block=False)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
mask = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
plt.imshow(mask, cmap="gray"), plt.show(block=False)


def calcular_factor_forma(img):
    ext_cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_cont[0])
    perimeter = cv2.arcLength(ext_cont[0], True)
    rho = 4 * np.pi * area / (perimeter**2)
    return rho


n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

masc_monedas = np.zeros_like(img)
masc_dados = np.zeros_like(img)
img_out = img

monedas = list()
dados = list()

RHO_TH = 0.8

for i in range(1, n):
    obj = (labels == i).astype("uint8") * 255

    coor_h = stats[i][cv2.CC_STAT_LEFT]
    coor_v = stats[i][cv2.CC_STAT_TOP]

    ancho = stats[i][cv2.CC_STAT_WIDTH]
    largo = stats[i][cv2.CC_STAT_HEIGHT]

    imagen = img_gray[coor_v : coor_v + largo, coor_h : coor_h + ancho]

    rho = calcular_factor_forma(obj)

    if rho >= RHO_TH:
        # masc_monedas = np.logical_or(obj, masc_monedas)
        masc_monedas[obj == 255, 2] = 255
        monedas.append((imagen, stats))
    else:
        # masc_dados = np.logical_or(obj, masc_dados)
        masc_monedas[obj == 255, 0] = 255
        dados.append((imagen, stats))


plt.imshow(masc_monedas, cmap="gray"), plt.show()

plt.imshow(cv2.add(img, masc_monedas)), plt.show()


plt.imshow(masc_dados, cmap="gray"), plt.show()

plt.imshow(dados[1][0], cmap="gray"), plt.show()

contar_circulos(dados[0][0])


plt.imshow((labels == 1).astype("uint8") * 255, cmap="gray")
plt.show()

plt.imshow(masc_dados, cmap="gray"), plt.show()


# Monedas y dados
plt.imshow(np.bitwise_and(mask, img_gray), cmap="gray"), plt.show()
# Monedas
plt.imshow(np.bitwise_and(masc_monedas * 255, img_gray), cmap="gray"), plt.show()
# Dados
plt.imshow(np.bitwise_and(masc_dados * 255, img_gray), cmap="gray"), plt.show()


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
kernel_re = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
kernel_cr = cv2.getStructuringElement(cv2.MORPH_CROSS, (40, 40))

img_morph = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
img_re = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel_re)
img_cr = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel_cr)

# ax = plt.subplot(121)
# plt.title('Original')
# plt.imshow(img_rgb, cmap='gray')
# plt.subplot(122, sharex=ax, sharey=ax)
# plt.title('Canny')
# plt.imshow(img_canny, cmap='gray')
# plt.show()

ax = plt.subplot(221)
plt.title("Canny")
plt.imshow(img_canny, cmap="gray")
plt.subplot(222, sharex=ax, sharey=ax)
plt.title("Ellipse")
plt.imshow(img_morph, cmap="gray")
plt.subplot(223, sharex=ax, sharey=ax)
plt.title("Rect")
plt.imshow(img_re, cmap="gray")
plt.subplot(224, sharex=ax, sharey=ax)
plt.title("Cross")
plt.imshow(img_cr, cmap="gray")
plt.show()

# Contamos cada objeto
n, labels, stats, _ = cv2.connectedComponentsWithStats(img_morph)

dados = list()
monedas = list()

AREA_TH = 5000
MONEDA_C = (255, 0, 0)
DADO_C = (0, 255, 0)

img_ejercicio_a = img_rgb.copy()

# Segmentamos entre monedas y dados
for i in range(1, n):
    if stats[i][-1] < AREA_TH:
        continue

    coor_h = stats[i][0]
    ancho = stats[i][2]

    coor_v = stats[i][1]
    largo = stats[i][3]

    imagen = img_gray[coor_v : coor_v + largo, coor_h : coor_h + ancho]

    cant_circulos = contar_circulos(imagen)

    if cant_circulos > 0:
        c = DADO_C
        dados.append((imagen, stats[i]))
    else:
        c = MONEDA_C
        monedas.append((imagen, stats[i]))

    graficar_caja(img_ejercicio_a, stats[i], c)

plt.imshow(img_ejercicio_a)
plt.title("Monedas y dados segementados")
plt.show()


# plt.subplot(131)
# plt.title(monedas[2][0].shape)
# plt.imshow(monedas[2][0], cmap='gray')
# plt.subplot(132)
# plt.title(monedas[3][0].shape)
# plt.imshow(monedas[3][0], cmap='gray')
# plt.subplot(133)
# plt.title(monedas[0][0].shape)
# plt.imshow(monedas[0][0], cmap='gray')
# plt.show()

MONEDAS_S_C = (255, 0, 0)
MONEDAS_M_C = (0, 255, 0)
MONEDAS_L_C = (0, 0, 255)

monedas_s = monedas_m = monedas_l = 0

img_ejercicio_b = img_rgb.copy()


chicas = medianas = grandes = 0
for moneda in monedas:
    ancho = moneda[0].shape[0]

    if ancho >= 360:
        grandes += 1
    elif ancho <= 320:
        chicas += 1
    else:
        medianas += 1

    plt.imshow(moneda[0], cmap="gray")
    plt.title(moneda[0].shape)
    plt.show()


for moneda in monedas:
    ancho = moneda[0].shape[0]
    # Monedas de 50 centavos, grandes
    if ancho >= 340:
        c = MONEDAS_L_C
        monedas_l += 1
    # Monedas de un peso, medianas
    elif 300 <= ancho <= 335:
        c = MONEDAS_M_C
        monedas_m += 1
    # El resto son las monedas chicas
    else:
        c = MONEDAS_S_C
        monedas_s += 1

    graficar_caja(img_ejercicio_b, moneda[1], c)

plt.imshow(img_ejercicio_b)
plt.title(f"Chicas: {monedas_s}, medianas: {monedas_m}, grandes: {monedas_l}")
plt.show()

# plt.subplot(121)
# plt.title(f'Número: {contar_circulos(dados[0][0])}')
# plt.imshow(dados[0][0], cmap='gray')
# plt.subplot(122)
# plt.title(f'Número: {contar_circulos(dados[1][0])}')
# plt.imshow(dados[1][0], cmap='gray')
# plt.show()

for dado in dados:
    plt.imshow(dado[0], cmap="gray")
    n = contar_circulos(dado[0])
    plt.title(f"Número: {n}")
    plt.show()
