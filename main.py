import cv2 as cv
import numpy as np
from copy import copy
import math

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# VIDEO_NAME = "video1.webm"
VIDEO_NAME = 0

class Video():
  def __init__(self):
    cv.namedWindow('image')
    cv.namedWindow('trackbars')
    self.create_trackbars()
    self.RED_VALUE = 100
    self.BLUE_VALUE = 100
    self.GREEN_VALUE = 100
    self.correct = False
    self.reset = False


    self.cap = cv.VideoCapture(VIDEO_NAME)
    self.canvas = self.create_canvas()

    self.last_pos = {
      "value" : None,
      "time" : 0
    }

#     while 1:
#       self.next()
#       cv.waitKey(1)
#     cv.destroyAllWindows()

  def next(self):
    pos = [0, 0]
    sucess, image = self.cap.read()
    if not sucess:
      print("not success")
      return

    self.reset = cv.getTrackbarPos('Reset', 'trackbars')
    if self.reset:
      self.canvas = self.create_canvas()
      self.reset = False

    image = cv.resize(image, (WIDTH, WIDTH))
    image = cv.flip(image, 1)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_hsv, higher_hsv = self.get_trackbar_position()
    mask = cv.inRange(hsv_image, lower_hsv, higher_hsv)
    frame = cv.bitwise_and(hsv_image, hsv_image, mask=mask)
    test = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
    last = copy(pos)
    test, pos = self.find_finger(test)

    if self.correct and pos is not None and last is not None:
      if abs(last[0] - pos[0]) < 15 and abs(last[1] - pos[1]) > 15:
        pos[0] = last[0]
      elif abs(last[1] - pos[1]) < 15 and abs(last[0] - pos[0]) > 15:
        pos[1] = last[1]

    self.canvas = self.draw_point(self.canvas, pos)
    # cv.imshow('image', frame)
    #cv.imshow('real', test)
    # cv.imshow('canvas', self.canvas)
    return frame, self.canvas


  def draw_point(self, frame, pos):
    frame = cv.circle(frame, pos, 3, (0, 0, 0), 4)
    return frame

  def find_finger(self, frame):
    rows, cols, _ = frame.shape
    new_frame = frame.copy()
    finish = False
    pos = None
    if self.last_pos["value"] is not None:
      dist_range = int(WIDTH * 0.3 / 2)
      hor_init = max(self.last_pos["value"][0] - dist_range, 0)
      ver_init = max(self.last_pos["value"][1] - dist_range, 0)
      hor_end = min(self.last_pos["value"][0] + dist_range, WIDTH)
      ver_end = min(self.last_pos["value"][1] + dist_range, WIDTH)
    else:
      hor_init = 0 
      hor_end = WIDTH
      ver_init = 0 
      ver_end = WIDTH
    for i in range(ver_init, ver_end):
      for j in range(hor_init, hor_end):
        p = frame[i,j]
        dist = 0
        if self.last_pos["value"] is not None and self.last_pos["time"] < 30:
          dist = math.sqrt(abs(i - self.last_pos["value"][0]) ** 2 + abs(j - self.last_pos["value"][1] ** 2))

        around = self.check_around(frame, [i, j])
        if around and dist < WIDTH * 0.3:
          new_frame = cv.circle(frame, [j, i], 4, (0, 255, 0), 1)
          pos = [j, i]
          finish = True
          break
      if finish:
       break

    if pos is not None:
      self.last_pos["value"] = copy(pos)
    else:
      self.last_pos["time"] += 1
      return frame, self.last_pos["value"]

    return frame, pos
  
  def check_around(self, frame, pos):
    x, y = pos[0], pos[1]
    finish = False
    if y >= WIDTH - 10 or x >= WIDTH - 1:
      return False
    for i in range(5):
      r = frame[x,y+i][2] < self.RED_VALUE or frame[x+1, y+i][2] < self.RED_VALUE or frame[x-1, y+i][2] < self.RED_VALUE
      g = frame[x,y+i][2] < self.GREEN_VALUE or frame[x+1, y+i][2] < self.GREEN_VALUE or frame[x-1, y+i][2] < self.GREEN_VALUE
      b = frame[x,y+i][2] < self.BLUE_VALUE or frame[x+1, y+i][2] < self.BLUE_VALUE or frame[x-1, y+i][2] < self.BLUE_VALUE
      if r and g and b:
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
    self.RED_VALUE = cv.getTrackbarPos('Red', 'trackbars')
    self.BLUE_VALUE = cv.getTrackbarPos('Blue', 'trackbars')
    self.GREEN_VALUE = cv.getTrackbarPos('Green', 'trackbars')
    self.correct = cv.getTrackbarPos('Correct', 'trackbars')
    lower_hsv = np.array([lh, ls, lv])
    higher_hsv = np.array([uh, us, uv])
    return lower_hsv, higher_hsv
    

  def create_trackbars(self):
    lh, ls, lv = 164, 71, 0
    uh, us, uv = 255, 165, 255
    cv.createTrackbar('Min_Hue', 'trackbars', lh, 255, Video.nothing)
    cv.createTrackbar('Min_Saturation', 'trackbars', ls, 255, Video.nothing)
    cv.createTrackbar('Min_Value', 'trackbars', lv, 255, Video.nothing)
    cv.createTrackbar('Max_Hue', 'trackbars', uh, 255, Video.nothing)
    cv.createTrackbar('Max_Saturation', 'trackbars', us, 255, Video.nothing)
    cv.createTrackbar('Max_Value', 'trackbars', uv, 255, Video.nothing)
    cv.createTrackbar('Red', 'trackbars', 100, 255, Video.nothing)
    cv.createTrackbar('Red', 'trackbars',100, 255, Video.nothing)
    cv.createTrackbar('Blue', 'trackbars', 100, 255, Video.nothing)
    cv.createTrackbar('Green', 'trackbars', 100, 255, Video.nothing)
    cv.createTrackbar('Correct', 'trackbars', 0, 1, Video.nothing)
    cv.createTrackbar('Reset', 'trackbars', 0, 1, Video.nothing)

  
  def create_canvas(self):
    blank_image = 255 * np.ones(shape=[WIDTH, WIDTH, 3], dtype=np.uint8)
    return blank_image
  
  def wipe_canvas(self):
    for i in range(len(self.canvas)):
      for a in range(len(self.canvas[0])):
        self.canvas[i][a] = [0, 0, 0];
  
  def nothing(hello):
    pass


class VideoApp(App):
  def build(self):
    print("Build")
    self.frame = Image()
    layout = BoxLayout() 
    layout.add_widget(self.frame)
    Clock.schedule_interval(self.update, 1 / 60)
    self.video = Video()

    return layout

  def update(self, dt):
    frame, canvas = self.video.next()
    buf = frame.tostring()
    texture = Texture.create(size=(WIDTH, WIDTH), colorfmt="bgr")
    texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
    self.frame.texture = texture


if __name__ == "__main__":
  WIDTH = 500
  videoapp = VideoApp().run()
