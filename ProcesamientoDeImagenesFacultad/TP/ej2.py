import cv2
import numpy as np
from math import dist
from argparse import ArgumentParser
from os.path import basename

# import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("Imgs", nargs="+", help="Path de las imágenes")

args = parser.parse_args()


def procesar_formulario(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    UMBRAL = 118
    MAX_COLS = 400
    MAX_ROWS = 800

    img_binaria = img < UMBRAL

    cols = np.sum(img_binaria, 0)
    rows = np.sum(img_binaria, 1)

    cols_idx = np.argwhere(cols > MAX_COLS)
    rows_idx = np.argwhere(rows > MAX_ROWS)

    _, c2, c3 = cols_idx[:, 0]
    _, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11 = rows_idx[:, 0]

    coordenadas = {
        "nombre": (r2, r3),
        "edad": (r3, r4),
        "mail": (r4, r5),
        "legajo": (r5, r6),
        "pregunta1": (r7, r8),
        "pregunta2": (r8, r9),
        "pregunta3": (r9, r10),
        "comentarios": (r10, r11),
    }

    img_dict = dict()

    for key, (start_row, end_row) in coordenadas.items():
        img_dict[key] = [img_binaria[start_row + 2 : end_row - 2, c2 + 2 : c3 - 2], 0]

    return img_dict


def contar_espacios(stats):
    stats_sort = stats[stats[:, 0].argsort()][1:, :]

    espacios = 0
    DISTANCIA_MAX = 9

    for i in range(len(stats_sort) - 1):
        primer_punto = (stats_sort[i][0] + stats_sort[i][2], 0)
        segundo_punto = (stats_sort[i + 1][0], 0)

        if dist(primer_punto, segundo_punto) >= DISTANCIA_MAX:
            espacios += 1

    return espacios


def connected_components(img):
    return cv2.connectedComponentsWithStats(img[0].astype(np.uint8), 8, cv2.CV_32S)


def validar_nombre(img_dict):
    num_labels, _, stats, _ = connected_components(img_dict["nombre"])
    espacios = contar_espacios(stats)

    if espacios > 0 and (num_labels + espacios - 1) <= 25:
        img_dict["nombre"][1] = 1


def validar_edad(img_dict):
    num_labels, _, stats, _ = connected_components(img_dict["edad"])
    espacios = contar_espacios(stats)

    if (num_labels - 1 + espacios) in [2, 3]:
        img_dict["edad"][1] = 1


def validar_mail(img_dict):
    num_labels, _, stats, _ = connected_components(img_dict["mail"])
    espacios = contar_espacios(stats)

    if num_labels - 1 and espacios == 0 and num_labels - 1 <= 25:
        img_dict["mail"][1] = 1


def validar_legajo(img_dict):
    num_labels, _, stats, _ = connected_components(img_dict["legajo"])
    espacios = contar_espacios(stats)

    if espacios == 0 and num_labels - 1 == 8:
        img_dict["legajo"][1] = 1


def validar_preguntas(img_dict):
    keys = [f"pregunta{i}" for i in range(1, 4)]

    for key in keys:
        num_l, *_ = connected_components(img_dict[key])

        if num_l == 3:
            img_dict[key][1] = 1


def validar_comentarios(img_dict):
    num_labels, _, stats, _ = connected_components(img_dict["comentarios"])
    espacios = contar_espacios(stats)

    if num_labels - 1 and (num_labels - 1 + espacios) <= 25:
        img_dict["comentarios"][1] = 1


def validar_imagen(img_dict):
    funciones = [
        validar_nombre,
        validar_edad,
        validar_mail,
        validar_legajo,
        validar_preguntas,
        validar_comentarios,
    ]

    for funcion in funciones:
        funcion(img_dict)

    for key, (_, val2) in img_dict.items():
        print(f'{key.capitalize() + ":":<20} {"OK" if val2 else "MAL"}')


for path in args.Imgs:
    img = procesar_formulario(path)

    if img:
        print(basename(path))
        print("=" * 24)
        validar_imagen(img)

    print()


# El programa utiliza las bibliotecas cv2 (OpenCV), numpy y argparse para procesar imágenes y manejar argumentos de línea de comandos.
# El propósito principal del código es procesar una o más imágenes de formularios escaneados,identificar campos específicos en los formularios
# y verificar si estos campos cumplen ciertos criterios de validez.

# El programa comienza definiendo un objeto ArgumentParser para manejar argumentos de línea de comandos. Toma una lista de rutas de
# imágenes como entrada utilizando el argumento "Imgs". El usuario proporciona estas rutas al ejecutar el programa.
# El código define una función llamada procesar_formulario(path) que toma la ruta de una imagen como entrada y procesa el formulario en esa imagen.
# Primero, carga la imagen en escala de grises utilizando cv2.imread.
# Luego, se aplica un umbral (UMBRAL) para convertir la imagen en binaria, donde los píxeles oscuros representan el contenido del formulario.
#
# El programa realiza recuentos de píxeles en las columnas y filas de la imagen binaria para identificar áreas significativas.
# Luego, determina las coordenadas de regiones específicas en el formulario, como el nombre, la edad, el correo electrónico, el legajo, las respuestas a preguntas y los comentarios.
# Estas coordenadas se almacenan en un diccionario llamado coordenadas.
#
# El programa crea un diccionario img_dict para almacenar las regiones del formulario y un valor asociado que indica si cada región cumple con ciertos criterios de validez.
# Luego, se invoca una serie de funciones de validación para cada región del formulario, como el nombre, la edad, el correo electrónico, el legajo, las preguntas y los comentarios.
# Si una región cumple con los criterios de validez, se establece el valor asociado en 1; de lo contrario, se mantiene en 0.
#
# El código principal del programa procesa cada imagen proporcionada por el usuario.
# Primero, llama a la función procesar_formulario para obtener el diccionario img_dict con las regiones y sus estados de validez.
# Luego, imprime el nombre de la imagen y muestra el resultado de la validación para cada región del formulario, indicando si cada región es "OK"
# (cumple con los criterios de validez) o "MAL" (no cumple con los criterios de validez).
#
# El programa se diseñó para procesar formularios específicos, donde se espera que las regiones del formulario tengan un diseño y una estructura predefinidos.
# Los criterios de validez se basan en la estructura esperada de los formularios, como el número de componentes conectados en ciertas regiones y la distancia entre ellos.
#
# En resumen, el código es un programa de procesamiento de imágenes que valida formularios escaneados.
# Utiliza técnicas de procesamiento de imágenes para identificar y evaluar regiones específicas en los formularios y determina si cumplen con los criterios de validez predefinidos.
# Esto se logra utilizando las bibliotecas OpenCV y numpy para el procesamiento de imágenes y argparse para manejar argumentos de línea de comandos.
# El programa es específico para un tipo particular de formulario y proporciona resultados de validación para cada región del formulario en función de los criterios establecidos.
