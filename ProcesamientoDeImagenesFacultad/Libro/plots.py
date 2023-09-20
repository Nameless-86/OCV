from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# read image to array
im = np.array(Image.open("foton.jpg"))
# plot the image
plt.imshow(im)
# some points
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]
# plot the points with red star-markers
plt.plot(x, y, "r*")
# line plot connecting the first two points
plt.plot(x[:2], y[:2])

plt.plot(x, y)  # default blue solid line
plt.plot(x, y, "r*")  # red star-markers
plt.plot(x, y, "go-")  # green line with circle-markers
plt.plot(x, y, "ks:")  # black dotted line with square-markers

# add title and show the plot
plt.title('Plotting: "foton.jpg"')
plt.show()
