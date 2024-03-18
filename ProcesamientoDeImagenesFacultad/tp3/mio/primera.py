import cv2
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt

NOMBRE = "./videos/tirada_1.mp4"
FRAME = 75


cap = cv2.VideoCapture(NOMBRE)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME - 1)
res, frame = cap.read()


# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
my_video_name = NOMBRE.split(".")[0]

# Display the resulting frame
cv2.imshow(my_video_name + " frame " + str(FRAME), frame)


# Store this frame to an image
cv2.imwrite(my_video_name + "_frame_" + str(FRAME) + ".jpg", frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
