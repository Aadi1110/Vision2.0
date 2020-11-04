import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

img = cv2.imread('testrun0.png',1)
ar = np.zeros((9,9), dtype=int)
arrx = np.empty((9,9), dtype=float)
arry = np.empty((9,9), dtype=float)
adj = np.zeros((81,81), dtype=int)
w = int(img.shape[1])
h = int(img.shape[0])
for i in range (1,10,1):
	for j in range (1,10,1):
		arrx[i-1,j-1] =  int(w*j/10)
		arry[i-1,j-1] = int(h*i/10)
		#cv2.circle(img, (int(w*i/10),int(h*j/10)), radius=0, color=(0,0,255), thickness=2)
for i in range(8):
	adj[i,i+1] = 1
for i in range(8,72,9):
	adj[i,i+9] = 1
for i in range(73,81,1):
	adj[i,i-1] = 1
for i in range(9,73,9):
	adj[i,i-9] = 1
adj[4,13] = 1
adj[36,37] = 1
adj[44,43] = 1
adj[76,67] = 1
for i in range(20,24,1):
	adj[i,i+1] = 1
for i in range(24,52,9):
	adj[i,i+9] = 1
for i in range(57,61,1):
	adj[i,i-1] = 1
for i in range(29,57,9):
	adj[i,i-9] = 1
adj[38,39] = 1
adj[38,37] = 1
adj[58,67] = 1
adj[58,49] = 1
adj[42,43] = 1
adj[42,41] = 1
adj[22,31] = 1
adj[22,13] = 1
adj[13,4] = 1
adj[13,22] = 1
adj[37,38] = 1
adj[37,36] = 1
adj[43,44] = 1
adj[43,42] = 1
adj[67,76] = 1
adj[67,58] = 1

lY = np.array([20,100,100])
uY = np.array([40,255,255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
maskY = cv2.inRange(hsv,lY,uY)
kernel = np.ones((1,1), np.uint8)
erY = cv2.erode(maskY, kernel, iterations = 1)
dilY = cv2.dilate(erY, kernel, iterations = 1)

contoursY,_ = cv2.findContours(dilY, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cfY1=[]
cfY2=[]
cfY3=[]
for cnt in contoursY:
	eps = 0.02*cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, eps, True)
	if(len(approx)==3):
		#cfY1.append((approx[0]+approx[1]+approx[2])/3)
		M = cv2.moments(approx)
		cx = (M['m10']/M['m00'])
		cy = (M['m01']/M['m00'])
		cfY1.append([cx,cy])
	elif(len(approx)==4):
		#cfY2.append((approx[0]+approx[1]+approx[2]+approx[3])/4)
		M = cv2.moments(approx)
		cx = (M['m10']/M['m00'])
		cy = (M['m01']/M['m00'])
		cfY2.append([cx,cy])
	else:
		M = cv2.moments(approx)
		cx = (M['m10']/M['m00'])
		cy = (M['m01']/M['m00'])
		cfY3.append([cx,cy])

for c in cfY1:
	#print(round(c[0][0]*10/w)-1,round(c[0][1]*10/h)-1)
	ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 4
for c in cfY2:
	ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 5
for c in cfY3:
	ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 6

lR = np.array([0,100,100])
uR = np.array([10,255,255])
maskR = cv2.inRange(hsv,lR,uR)
erR = cv2.erode(maskR, kernel, iterations = 1)
dilR = cv2.dilate(erR, kernel, iterations = 1)
contoursR,_ = cv2.findContours(dilR, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cfR1=[]
cfR2=[]
cfR3=[]
for cnt in contoursR:
	eps = 0.02*cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, eps, True)
	if(len(approx)==3):
		#cfR1.append((approx[0]+approx[1]+approx[2])/3)
		M = cv2.moments(approx)
		cx = (M['m10']/M['m00'])
		cy = (M['m01']/M['m00'])
		cfR1.append([cx,cy])
	elif(len(approx)==4):
		#cfR2.append((approx[0]+approx[1]+approx[2]+approx[3])/4)
		M = cv2.moments(approx)
		cx = (M['m10']/M['m00'])
		cy = (M['m01']/M['m00'])
		cfR2.append([cx,cy])
	else:
		M = cv2.moments(approx)
		cx = (M['m10']/M['m00'])
		cy = (M['m01']/M['m00'])
		cfR3.append([cx,cy])

for c in cfR1:
	ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 1
for c in cfR2:
	ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 2
for c in cfR3:
	ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 3

print(ar)

#plt.imshow(img)
#plt.show()
#print(adj)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows();