from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = np.array(Image.open("foton.jpg"))
plt.imshow(im)
print("Please click 3 points")
x = plt.ginput(3)
print("you clicked:", x)
plt.show()
