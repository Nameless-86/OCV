import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank', blank)

# Paint image

# blank[:] = 0, 255, 0
# cv.imshow('Green', blank)
# blank[200:300, 300:400] = 0, 0, 255
# cv.imshow('Red Square', blank)

# Draw rectangles

cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=2)

################################################################
# cv.rectangle(blank, (0, 0),
#              (blank.shape[1]//2, blank.shape[0]//2,), (255, 0, 0), thickness=cv.FILLED)
# cv.imshow('Rectangle', blank)

# Draw circles

# cv.circle(blank, (blank.shape[1]//2, blank.shape[0] //
#           2,), 40, (155, 255, 0), thickness=-1)
# cv.imshow('Circle', blank)


# Draw lines

cv.line(blank, (0, 0), (250, 250), (255, 255, 255), thickness=3)
cv.imshow('Line', blank)


# Write Text

cv.putText(blank, 'Hello', (255, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (123, 0, 213), thickness=1)
cv.imshow('Text', blank)


# img = cv.imread('Photos/cat.jpg')
# cv.imshow('Cat', img)

cv.waitKey(0)
