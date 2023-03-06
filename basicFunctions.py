import cv2 as cv

img = cv.imread('Photos/park.jpg')

cv.imshow('Original', img)

## Convert to gray scale ##
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

## Blur an image ##
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

## Edge Cascade ##
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

## Dilating the image ##
dilated = cv.dilate(img, (3, 3), iterations=1)
cv.imshow('Dilating', dilated)

## Eroding the image ##
eroded = cv.erode(img, (7, 7), iterations=3)
cv.imshow('Eroded', eroded)

## Resize the image ##
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

## Cropping the image ##
cropped = img[50:200,200:400]
cv.imshow('Cropped', cropped)


cv.waitKey(0)
