import cv2
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("./videos/tirada_1.mp4")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(width, height, fps, n_frames)

fig, axs = plt.subplots(4, 4, figsize=(20, 20))
axs = axs.flatten()

img_idx = 0
for frame in range(n_frames):
    ret, img = cap.read()
    if ret == False:
        break
    if frame % 10 == 0:
        axs[img_idx].imshow(img)
        axs[img_idx].set_title(f"frame: {frame}")
        axs[img_idx].axis("off")
        img_idx += 1

plt.tight_layout()
plt.show()
cap.release()
cv2.destroyAllWindows()
