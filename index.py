import cv2 as cv

def nothing():
  pass

cv.namedWindow('image')
cv.createTrackbar('lowH','image',0,179,nothing)
cv.createTrackbar('highH','image',179,179,nothing)

cv.createTrackbar('lowS','image',0,255,nothing)
cv.createTrackbar('highS','image',255,255,nothing)

cv.createTrackbar('lowV','image',0,255,nothing)
cv.createTrackbar('highV','image',255,255,nothing)
image = cv.imread("proxy-image.jpeg")
cv.imshow("image", image)

hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)


cv.waitKey(0)
# and finally destroy/close all open windows
cv.destroyAllWindows()

