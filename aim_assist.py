# aim assist is a program that will automatically aim at a bot in a game.
#imports
import cv2 as cv
from time import time
from screen_capture import ScreenCapture
from vision import Vision

# ScreenCapture.list_window_names()
# exit()

screencap = ScreenCapture('Scrap Mechanic')
# initialize the vision class
vision_class = Vision(None)
bot_vision = cv.CascadeClassifier('cascade/cascade.xml')

loop_time = time()
# loop time
while (True):
    screenshot = screencap.get_screenshot()
    
    # do object detection
    rectangles = bot_vision.detectMultiScale(screenshot)

    # find the crosshairs
    detection_image = vision_class.draw_rectangles(screenshot, rectangles)

    # display the images
    #  small image 
    small_image = cv.resize(detection_image, (0, 0), fx=0.5, fy=0.5)
    cv.imshow('Matches', small_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # press 'f' to save screenshot as a positive image, press 'd' to 
    # save as a negative image.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    # esc
    if key == 27:
        cv.destroyAllWindows()
        break
    elif key == ord('f'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
    elif key == ord('d'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)

print("Done.")






