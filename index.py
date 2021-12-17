import cv2 as cv
import numpy as np

VIDEO_NAME = "video.mp4"

class Video():
  def __init__(self):
    cv.namedWindow('image')
    cv.namedWindow('trackbars')
    self.create_trackbars()

    cap = cv.VideoCapture(VIDEO_NAME)

    while 1:
      cap = cv.VideoCapture(VIDEO_NAME)
      sucess, image = cap.read()

      while sucess:
        # image = cv.resize(image, [256, 256])
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower_hsv, higher_hsv = self.get_trackbar_position()
        mask = cv.inRange(hsv_image, lower_hsv, higher_hsv)
        frame = cv.bitwise_and(hsv_image, hsv_image, mask=mask)
        cv.imshow('image', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
          break
        sucess, image = cap.read()
    cv.destroyAllWindows()

  def get_trackbar_position(self):
    lh = cv.getTrackbarPos('Min_Hue', 'trackbars')
    ls = cv.getTrackbarPos('Min_Saturation', 'trackbars')
    lv = cv.getTrackbarPos('Min_Value', 'trackbars')
    uh = cv.getTrackbarPos('Max_Hue', 'trackbars')
    us = cv.getTrackbarPos('Max_Saturation', 'trackbars')
    uv = cv.getTrackbarPos('Max_Value', 'trackbars')
    lower_hsv = np.array([lh, ls, lv])
    higher_hsv = np.array([uh, us, uv])
    return lower_hsv, higher_hsv
    

  def create_trackbars(self):
    lh, ls, lv = 1, 81, 109
    uh, us, uv = 12, 180, 250
    cv.createTrackbar('Min_Hue', 'trackbars', lh, 255, Video.nothing)
    cv.createTrackbar('Min_Saturation', 'trackbars', ls, 255, Video.nothing)
    cv.createTrackbar('Min_Value', 'trackbars', lv, 255, Video.nothing)
    cv.createTrackbar('Max_Hue', 'trackbars', uh, 255, Video.nothing)
    cv.createTrackbar('Max_Saturation', 'trackbars', us, 255, Video.nothing)
    cv.createTrackbar('Max_Value', 'trackbars', uv, 255, Video.nothing)
  
  def nothing(hello):
    pass



Video()


