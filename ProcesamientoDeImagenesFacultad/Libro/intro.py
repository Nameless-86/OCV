from PIL import Image

pil_im = Image.open("foton.jpg")
pil_im_c = Image.open("foton.jpg").convert("L")

print(pil_im)
