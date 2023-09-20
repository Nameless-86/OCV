from PIL import Image

pil_im = Image.open("foton.jpg")

box = (100, 100, 400, 400)

region = pil_im.crop(box)


region_t = region.transpose(Image.ROTATE_180)
pil_im.paste(region, box)
