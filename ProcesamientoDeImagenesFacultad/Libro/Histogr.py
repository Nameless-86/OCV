from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# read image to array
im = np.array(Image.open("foton.jpg").convert("L"))

# create new figure
plt.figure()

# dont use colors
plt.gray()

# Show contours with origin upper left corner
plt.contour(im, origin="image")
plt.axis("equal")
plt.axis("off")

plt.figure()
plt.hist(im.flatten(), 128)
plt.show()
