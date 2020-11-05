import cv2.aruco as aruco
import cv2
import math
import collections

class Graph:
	def __init__(self):
		self.graph = collections.defaultdict(list)
	def addEdge(self,u,v):
		self.graph[u].append(v)
	def getNeighbors(self,node):
		return self.graph[node]
	def changeLink(self,n1,n2):
		self.graph[n1] = []
		self.graph[n2] = []

def arucoRead():
	ARUCO_PARAMETERS = aruco.DetectorParameters_create()
	ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
	board = aruco.GridBoard_create(
	        markersX=2,
	        markersY=2,
	        markerLength=0.09,
	        markerSeparation=0.01,
	        dictionary=ARUCO_DICT)

	#img=cv2.imread('testrun0.png')
	img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	w = int(img.shape[1])
	h = int(img.shape[0])

	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

	center = (corners[0][0][0]+corners[0][0][1]+corners[0][0][2]+corners[0][0][3])/4
	grid = round(center[1]*10/h)*9+round(center[0]*10/w)
	slope = (corners[0][0][0][1]-corners[0][0][3][1])/(corners[0][0][0][0]-corners[0][0][3][0])
	angle = math.degrees(math.atan(slope))
	return(center, angle, grid)

def move(toGrid):
	cenGrid = [(toGrid%9+1)*h/10, (toGrid//9+1)*w/10]
	while True:
		cenBot, angBot,_ = arucoRead()
		slope = (cenBot[1]-cenGrid[1])/(cenBot[0]-cenGrid[0])
		angGrid = math.degrees(math.atan(slope))
		if(abs(angGrid-angBot)<7):
			#move_forward
		elif(angGrid-angBot>=7):
			#turn_left
		elif(angBot-angGrid>=7):
			#turn_right
		#stop
		if(abs(sum(cenBot-cenGrid))<10):
			#stop
			break
	return

def bfs(graph, start, goal):
    explored = []
    queue = [[start]]
 
    if start == goal:
        return [start]
 
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in explored:
            neighbours = graph.getNeighbors(node)
            
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                if neighbour == goal:
                    return new_path
 
            explored.append(node)
 
    return []

g = Graph()
for i in range(8):
	g.addEdge(i,i+1)
for i in range(8,72,9):
	g.addEdge(i,i+9)
for i in range(73,81,1):
	g.addEdge(i,i-1)
for i in range(9,73,9):
	g.addEdge(i,i-9)
for i in range(20,24,1):
	g.addEdge(i,i+1)
for i in range(24,52,9):
	g.addEdge(i,i+9)
for i in range(57,61,1):
	g.addEdge(i,i-1)
for i in range(29,57,9):
	g.addEdge(i,i-9)
g.addEdge(4,13)
g.addEdge(36,37)
g.addEdge(44,43)
g.addEdge(76,67)
#g.addEdge(38,39)
g.addEdge(38,37)
g.addEdge(58,67)
#g.addEdge(58,49)
g.addEdge(42,43)
#g.addEdge(42,41)
#g.addEdge(22,31)
g.addEdge(22,13)
g.addEdge(13,4)
g.addEdge(13,22)
g.addEdge(37,38)
g.addEdge(37,36)
g.addEdge(43,44)
g.addEdge(43,42)
g.addEdge(67,76)
g.addEdge(67,58)

cap = cv2.VideoCapture(0)
while(True):
	ret, img = cap.read()
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

	ar = np.zeros((9,9), dtype=int)

	_,_,grid = arucoRead()
	n1 = -1
	n2 = -1
	home = -1

	if(grid==36):
		n1 = 27
		n2 = 29
		home = 39

	elif(grid==44):
		n1 = 53
		n2 = 51
		home = 41

	elif(grid==4):
		n1 = 5
		n2 = 23
		home = 31

	elif(grid==76):
		n1 = 77
		n2 = 59
		home = 49

	counter = 0
	while True:
		_,_,start = arucoRead()
		val = int(input("Roll the die"))
		print("The value is: ",val)
		ends = []
		minpath = []
		l = 100
		for i in range(8):
			for j in range(8):
				if(ar[i][j]==val):
					ends.append(j*9+i)

		for node in ends:
			path = bfs(g,start,end)
			if(len(path>=2 and len(path)<l):
				minpath = path
				l = len(path)

		for i in range(1,len(minpath)):
			move(minpath[i])

