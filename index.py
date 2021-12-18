import cv2 as cv
import numpy as np
from copy import copy

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
      canvas = self.create_canvas()

      self.last_pos = {
        "value" : None,
        "time" : 0
      }

      while sucess:
        image = cv.resize(image, [256, 256])
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        lower_hsv, higher_hsv = self.get_trackbar_position()
        mask = cv.inRange(hsv_image, lower_hsv, higher_hsv)
        frame = cv.bitwise_and(hsv_image, hsv_image, mask=mask)
        test = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
        test, pos = self.find_finger(test)
        #canvas[pos[1]][pos[0]] = [0, 0, 0]
        canvas = self.draw_point(canvas, pos)
        cv.imshow('image', frame)
        cv.imshow('real', test)
        cv.imshow('canvas', canvas)

        if cv.waitKey(1) & 0xFF == ord('q'):
          break
        sucess, image = cap.read()
    cv.destroyAllWindows()

  def draw_point(self, frame, pos):
    frame = cv.circle(frame, pos, 3, (0, 0, 0), 4)
    return frame

  def find_finger(self, frame):
    rows, cols, _ = frame.shape
    new_frame = frame.copy()
    finish = False
    pos = [0, 0]
    if self.last_pos["value"] is not None:
      hor_init = max(self.last_pos["value"][0] - 75, 0)
      ver_init = max(self.last_pos["value"][1] - 75, 0)
      hor_end = min(self.last_pos["value"][0] + 75, 256)
      ver_end = min(self.last_pos["value"][1] + 75, 256)
    else:
      hor_init = 0 
      hor_end = 256
      ver_init = 0 
      ver_end = 256
    for i in range(ver_init, ver_end):
      for j in range(hor_init, hor_end):
        p = frame[i,j]
        dist = 0
        if self.last_pos["value"] is not None and self.last_pos["time"] < 30:
          dist = abs(i - self.last_pos["value"][0]) + abs(j - self.last_pos["value"][1]) 

        around = self.check_around(frame, [i, j])
        #if p[0] > 100 and frame[i,j-1][0] > 100 and dist < 200:
        if around and dist < 150:
          new_frame = cv.circle(frame, [j, i], 4, (0, 255, 0), 1)
          pos = [j, i]
          finish = True
          break
      if finish:
       break

    if pos is not None:
      self.last_pos = {
        "value": copy(pos),
        "time": 0
      }
    else:
      self.last_pos["time"] += 1
      return frame, self.last_pos.value

    return frame, pos
  
  def check_around(self, frame, pos):
    x, y = pos[0], pos[1]
    finish = False
    if y >= 256 - 10 or x >= 256 - 1:
      return False
    for i in range(5):
      if frame[x,y+i][0] < 100 or frame[x+1, y+i][0] < 100:
        finish = True
      if finish:
        break
    return not finish
          
        
    

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
  
  def create_canvas(self):
    blank_image = 255 * np.ones(shape=[256, 256, 3], dtype=np.uint8)
    return blank_image
  
  def nothing(hello):
    pass



Video()


