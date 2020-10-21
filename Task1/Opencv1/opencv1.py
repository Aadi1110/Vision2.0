import cv2
import numpy as np

img = cv2.imread('opencv1.jpeg',1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lg = np.array([35,150,0])
ug = np.array([80,255,255])
maskg = cv2.inRange(hsv,lg,ug)
maskg = cv2.GaussianBlur(maskg, (0,0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
green = cv2.bitwise_and(img,img,mask=maskg)
green[maskg==0] = (255,255,255)
cv2.imshow('green',green)
cv2.imwrite('green.jpg',green)

lb = np.array([80,50,50])
ub = np.array([150,255,255])
maskb = cv2.inRange(hsv,lb,ub)
maskb = cv2.GaussianBlur(maskb, (0,0), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_DEFAULT)
blue = cv2.bitwise_and(img,img,mask=maskb)
blue[maskb==0] = (255,255,255)
cv2.imshow('blue',blue)
cv2.imwrite('blue.jpg',blue)

lr = np.array([0,70,50])
ur = np.array([10,255,255])
maskr = cv2.inRange(hsv,lr,ur)
maskr = cv2.GaussianBlur(maskr, (0,0), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_DEFAULT)
red = cv2.bitwise_and(img,img,mask=maskr)
red[maskr==0] = (255,255,255)
cv2.imshow('red',red)
cv2.imwrite('red.jpg',red)

cv2.waitKey(0)
cv2.destroyAllWindows()
