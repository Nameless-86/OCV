from PIL import Image

pil_im = Image.open("foton.jpg")

out = pil_im.resize((128, 128))

out2 = pil_im.rotate(45)
