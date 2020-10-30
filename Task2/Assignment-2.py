import cv2
import numpy as np
scale_percent = 100

cap = cv2.VideoCapture('http://192.168.1.4:4747/video')
#cap = cv2.VideoCapture(0)
while True:
	ret,img = cap.read()
	width = int(img.shape[1]*scale_percent/100)
	height = int(img.shape[0]*scale_percent/100)
	dsize = (width, height)
	rsz = cv2.resize(img, dsize)
	hsv = cv2.cvtColor(rsz, cv2.COLOR_BGR2HSV)
	lY = np.array([20,100,100])
	uY = np.array([40,255,255])
	maskY = cv2.inRange(hsv,lY,uY)
	maskY = cv2.GaussianBlur(maskY, (0,0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
	kernel = np.ones((1,1), np.uint8)
	er = cv2.erode(maskY, kernel, iterations = 2)
	dil = cv2.dilate(er, kernel, iterations = 2)
	yel = cv2.bitwise_and(rsz,rsz,mask=dil)
	contours, hierarchy = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cf = []
	cd = []
	for cnt in contours:
		eps = 0.1*cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, eps, True)
		if(len(approx)==3 and cv2.contourArea(cnt)>=1500):
			cf.append(approx)
		else:
			cd.append(cnt)
	cv2.drawContours(yel, cf, -1, (0,255,0), 3)
	cv2.drawContours(yel, cd, -1, (0,0,0), -1)
	cv2.imshow('yel',yel)
	cv2.imshow('rsz',img)
	if ret==False:
		break
	k = cv2.waitKey(1)
	if(k==ord('q')):
		cap.release()
		cv2.destroyAllWindows()
		break