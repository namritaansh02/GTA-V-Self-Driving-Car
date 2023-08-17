from cv2 import bitwise_and
from grpc import intercept_channel
import numpy as np
from PIL import ImageGrab
import cv2
import time
from test_lanenet import test_lanenet
import matplotlib.pyplot as plt

# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)

# def debug_keypress(request):    
#     while(request):
#         print('down')
#         PressKey(W)
#         time.sleep(3)
#         print('up')
#         ReleaseKey(W)

# def make_coordinates(image, line_parameters):
#     try:
#         slope, intercept = line_parameters
#         y1 = int(image.shape[0])
#         y2 = int(y1*(3/5))
#         if slope != 0:
#             x1 = int((y1-intercept)//slope)
#             x2 = int((y2-intercept)//slope)
#         else :
#             x1 = int(0) 
#             x2 = int(0)
#         return np.array([x1, y1, x2, y2])
#     except TypeError:
#         return np.array([int(0), int(0), int(0), int(0)])

# def average(image, lines):
#     left_fit = []
#     right_fit = []
#     if lines is not None:
#         for line in lines: 
#             x1, y1, x2, y2 = line.reshape(4)
#             if x1 == x2:
#                 continue
#             parameters = np.polyfit((x1, x2), (y1, y2), 1)
#             slope = parameters[0]
#             intercept = parameters[1]
#             if slope<0:
#                 left_fit.append((slope, intercept))
#             else:
#                 right_fit.append((slope, intercept))
#     left_fit_average = np.average(left_fit, axis = 0)
#     right_fit_average = np.average(right_fit, axis = 0)
#     if left_fit_average:
#         left_line = make_coordinates(image, left_fit_average)
#     if right_fit_average:
#         right_line = make_coordinates(image, right_fit_average)
#     return np.array([left_line, right_line])

# def draw_lines(img, lines):
#     if lines is not None:
#         for line in lines:
#             print(line)
#             x1, y1, x2, y2 = line
#             cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), [255,255,255], 10)  

def process_image(original_image):  
    mask = np.zeros(original_image.shape, dtype = 'uint8')
    points = np.array([[0,500],[0,400],[200,300],[600,300],[800,400],[800,500]])
    cv2.fillPoly(mask, [points], (255,255,255))
    processed_image = bitwise_and(mask, original_image)
    return processed_image

last_time = time.time()

while(True):
    # screen = cv2.imread('./data/tusimple_test_image/0.jpg', cv2.IMREAD_COLOR)
    image = np.array(ImageGrab.grab(bbox=(1024,400,2048,720)))
    # image = process_image(image)
    scr1, scr2, scr3 = test_lanenet(image)

    print('Loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
      
    plt.figure('binary seg image')
    plt.imshow(scr1)
    plt.imshow(scr2)
    plt.imshow(scr3, cmap='gray')
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break 