import cv2 as cv
import numpy as np

img = cv.imread('Photos/cats.jpg')

cv.imshow('Cats', img)

blank = np.zeros(img.shape,dtype='uint8')
cv.imshow('Blanks', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Cats', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

canny = cv.Canny(img, 125, 175)
cv.imshow('Canny', canny)

# ret, tresh = cv.threshold(gray, 125,255, cv.THRESH_BINARY)
# cv.imshow('Threshold', tresh)

contours, hierarchies = cv.findContours(
    canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

print(f'{len(contours)} contours found!')

cv.drawContours(blank, contours, -1, (0,0,255),1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)
