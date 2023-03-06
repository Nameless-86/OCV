import cv2 as cv

img = cv.imread('Photos/cat.jpg')


def rescaleFrame(frame, scale=0.75):
    #Works for anything
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_image = rescaleFrame(img)
cv.imshow('Image', resized_image)

def changeRes(width,height):
    #Only works with live video
    capture.set(3,width)
    capture.set(4,height)

################Read Video ########################

capture = cv.VideoCapture('Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, scale=.2)

    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)
