import diplib as dip
import cv2 as cv
import numpy as np
from copy import copy
import skimage





def process(image):
    lower = np.array([0, 58, 53], dtype = "uint8")
    upper = np.array([30, 255, 255], dtype = "uint8")

    # lower2 = np.array([172, 30, 53], dtype = "uint8")
    # upper2 = np.array([180, 180, 210], dtype = "uint8")

    converted = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    skinMask = cv.inRange(converted, lower, upper)
    # skinMask2 = cv.inRange(converted, lower2, upper2)

    #Gaussian Blur
    skinMask = cv.GaussianBlur(skinMask, (7, 7), 0)
    # skinMask2 = cv.GaussianBlur(skinMask2, (3, 3), 0)

    skin = cv.bitwise_and(image, image, mask = skinMask)
    # skin2 = cv.bitwise_and(image, image, mask = skinMask2)
    # skin = cv.bitwise_or(skin1,skin2) #adding both ranges

    # show the skin in the image along with the mask
    ret, skin = cv.threshold(skin, 1, 255, cv.THRESH_BINARY)

    skin = cv.cvtColor(skin, cv.COLOR_BGR2GRAY)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(skin)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1
    min_size = 250

    im_result = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            im_result[im_with_separated_blobs == blob + 1] = 255

    im_result = im_result.astype(np.uint8)

    cv.imshow("images", np.hstack([image, im_result]))
    return im_result


if __name__ == "__main__":
    img_counter = 0
    cam = cv.VideoCapture(0)
    cv.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        y = 0
        x = 0
        h = 300
        w = 300
        process(frame[y:y+h, x:x+w])


        k = cv.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            GROUP = 4
            img_name = "images/{}/opencv_frame_{}.png".format(GROUP, img_counter)
            cv.imwrite(img_name, process(frame[y:y+h, x:x+w]))
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv.destroyAllWindows()
