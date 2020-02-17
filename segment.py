import cv2 as cv
import numpy as np

# 输入图片路径
car = cv.imread('images/hp recorder/IMG_009.JPG', 1)

cv.imshow('car', car)

# 设置不同种类车牌的颜色阈值
lower_blue = np.array([122, 50, 50])
upper_blue = np.array([130, 70, 90])
lower_yellow = np.array([22, 90, 55])
upper_yellow = np.array([112, 190, 100])
lower_white = np.array([0, 0, 205])
upper_white = np.array([50, 60, 255])

# 通过设置的阈值分割图像
car_hsv = cv.cvtColor(car, cv.COLOR_BGR2HSV)
mask_blue = cv.inRange(car_hsv, lower_blue, upper_blue)
mask_yellow = cv.inRange(car_hsv, lower_yellow, upper_yellow)
mask_white = cv.inRange(car_hsv, lower_white, upper_white)

# 将不同颜色作为分割阈值
mask_plate = cv.bitwise_and(car_hsv, car_hsv, mask=mask_blue+mask_yellow)
mask_gray = cv.cvtColor(mask_plate, cv.COLOR_BGR2GRAY)

# 利用形态学运算及二值分割提取车牌区域
m = np.ones((20, 20), np.uint8)
mask_close = cv.morphologyEx(mask_gray, cv.MORPH_CLOSE, m)
mask_open = cv.morphologyEx(mask_close, cv.MORPH_OPEN, m)
ret, mask = cv.threshold(mask_open, 0, 255, cv.THRESH_BINARY)

# 绘制分割边缘
area_list = []
# 设置车牌大小的左界和右界
area_range = [500, 10000]
contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    if area_range[0] < area < area_range[1]:
        area_list.append(cnt)
segment = cv.drawContours(car, area_list, -1, (0, 255, 0), 2)
segment = cv.resize(segment, (1000, 600))
cv.imshow('segment', segment)

cv.waitKey(0)