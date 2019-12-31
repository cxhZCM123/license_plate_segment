import cv2 as cv
import numpy as np

# 输入图片路径
car = cv.imread('images/hp recorder/IMG_017.JPG', 1)

cv.imshow('car', car)

# 设置蓝色与黄色的颜色阈值
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])
lower_yellow = np.array([15, 55, 55])
upper_yellow = np.array([50, 255, 255])

# 通过设置的阈值分割图像
car_hsv = cv.cvtColor(car, cv.COLOR_BGR2HSV)
mask_blue = cv.inRange(car_hsv, lower_blue, upper_blue)
mask_yellow = cv.inRange(car_hsv, lower_yellow, upper_yellow)
mask_plate = cv.bitwise_and(car_hsv, car_hsv, mask=mask_blue)
mask_gray = cv.cvtColor(mask_plate, cv.COLOR_BGR2GRAY)

# 利用形态学运算及二值分割提取车牌区域
Matrix = np.ones((20, 20), np.uint8)
mask_close = cv.morphologyEx(mask_gray, cv.MORPH_CLOSE, Matrix)
mask_open = cv.morphologyEx(mask_close, cv.MORPH_OPEN, Matrix)
ret, mask = cv.threshold(mask_open, 0, 255, cv.THRESH_BINARY)

# 绘制分割边缘
contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
segment = cv.drawContours(car, contours, -1, (0, 255, 0), 2)
segment = cv.resize(segment, (int(car.shape[1]/4), int(car.shape[0]/4)))
cv.imshow('segment', segment)
cv.waitKey(0)