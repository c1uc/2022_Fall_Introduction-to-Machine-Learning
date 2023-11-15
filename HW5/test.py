import cv2
import numpy as np


img1 = cv2.imread("./test/task3/1tFHZ1wB8xsOJjtV.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow('img1', img1)

_, img1 = cv2.threshold(img1, 220, 255, cv2.THRESH_BINARY)
cv2.imshow('bw', img1)

img1 = cv2.bitwise_not(img1)
kernel = np.ones((3, 2), np.uint8)

img2 = cv2.erode(img1, kernel, iterations=1)
img2 = cv2.dilate(img2, kernel, iterations=1)
img2 = cv2.bitwise_not(img2)
cv2.imshow('img2', img2)
cv2.waitKey(0)
