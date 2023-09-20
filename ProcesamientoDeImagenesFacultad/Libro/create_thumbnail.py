from PIL import Image

pil_im = Image.open("foton.jpg")

pil_im.thumbnail((128, 128))
