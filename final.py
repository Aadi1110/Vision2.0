import cv2.aruco as aruco
import cv2
import math
import collections
import gym
import vision_arena
import numpy as np
import time
import pybullet as p
import sys

env = gym.make("vision_arena-v0")
img = env.camera_feed(is_flat=True)
# r = cv2.selectROI(feed)

# img = feed[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

####################################################################### Aruco 

def arucoRead():
    img = env.camera_feed(is_flat=True)
    p.stepSimulation()
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    board = aruco.GridBoard_create(markersX=2,markersY=2,markerLength=0.09,markerSeparation=0.01,dictionary=ARUCO_DICT)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = int(img.shape[1])
    h = int(img.shape[0])

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    center = np.array((corners[0][0][0]+corners[0][0][1]+corners[0][0][2]+corners[0][0][3])/4)
    grid = round(center[1]*10/h-1)*9+round(center[0]*10/w-1)
    #slope = (corners[0][0][0][1]-corners[0][0][3][1])/(corners[0][0][0][0]-corners[0][0][3][0])
    bvec = np.array([(corners[0][0][0][0]-corners[0][0][3][0]),(corners[0][0][0][1]-corners[0][0][3][1])])
    
    return(center, bvec, grid)

####################################################################### Move Bot Function

s2 = 350
s1 = 250
s180 = 80
ka = 0.8
kt = 1.0
def dist(p1,p2):
    d=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return d

def ang(v,bvec):
    #print(bvec,v)
    c1 = complex(bvec[0],bvec[1])
    c2 = complex(v[0],v[1])
    angle = np.angle(c1/c2)
    angle = math.degrees(angle)
    return angle

def rotate(tovec):
    while True:
        cenBot, bvec,pos = arucoRead()
        v = tovec - cenBot
        angle = ang(v,bvec)
        sr = int(s1 - ka*(90-abs(angle)))
        #print("Angle is: ",angle)
        if(dist(tovec,cenBot)>15):
            if(abs(angle)<=10):
                break
            if(angle>0):
                if(angle>90):
                    for i in range(sr):
                        p.stepSimulation()
                        env.move_husky(-1,1,-1,1)
                    #print("Left")
                else:
                    for i in range(sr):
                        p.stepSimulation()
                        env.move_husky(-0.6,1,-0.6,1)
                    #print("Left")
                	#time.sleep(0.01)
            else:
                if(angle<-90):
                    for i in range(sr):
                        p.stepSimulation()
                        env.move_husky(1,-1,1,-1)
                    #print("Right")
                else:
                    for i in range(sr):
                        p.stepSimulation()
                        env.move_husky(1,-0.6,1,-0.6)
                    #print("Right")
                    #time.sleep(0.01)
        else:
            for i in range(s1):
                p.stepSimulation()
                env.move_husky(0,0,0,0)
            break


def move(toGrid):
    
    tovec = np.array([(toGrid%9+1)*h/10, (toGrid//9+1)*w/10])
    if(toGrid>=0 and toGrid<9):
        tovec[1] = tovec[1]-7
    if(toGrid%9==0):
        tovec[0] = tovec[0]-7
    img = env.camera_feed(is_flat=True)
    f=1
    while True:
        cenBot, bvec,pos = arucoRead()
        d = dist(tovec,cenBot)
        st = int(s2 - kt*(60-d))
        if(d<15):
        	break
        rotate(tovec)
        for i in range(st):
            p.stepSimulation()
            env.move_husky(1,1,1,1)
        #print("Forward")
        #time.sleep(0.01)


############################################################################ Graph and Adjacency Matrix

class Graph:
    def __init__(self):
        self.graph = collections.defaultdict(list)
    def addEdge(self,u,v):
        self.graph[u].append(v)
    def getNeighbors(self,node):
        return self.graph[node]
    def emptyLink(self,n1,n2):
        self.graph[n1] = []
        self.graph[n2] = []

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

################################################################################## simulation

ar = np.zeros((9,9), dtype=int)
arrx = np.empty((9,9), dtype=float)
arry = np.empty((9,9), dtype=float)
adj = np.zeros((81,81), dtype=int)

img = env.camera_feed(is_flat=True)
    
w = int(img.shape[1])
h = int(img.shape[0])

def extract():
    
    ############################################### Yellow Shapes Extraction
    
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
            if M['m00']!=0:
                cx = (M['m10']/M['m00'])
                cy = (M['m01']/M['m00'])
                cfY1.append([cx,cy])
        elif(len(approx)==4):
            #cfY2.append((approx[0]+approx[1]+approx[2]+approx[3])/4)
            M = cv2.moments(approx)
            if M['m00']!=0:
                cx = (M['m10']/M['m00'])
                cy = (M['m01']/M['m00'])
                cfY2.append([cx,cy])
        else:
            M = cv2.moments(approx)
            if M['m00']!=0:
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
        
    ############################################# Red Shapes 

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
            if M['m00']!=0:
                cx = (M['m10']/M['m00'])
                cy = (M['m01']/M['m00'])
                cfR1.append([cx,cy])
        elif(len(approx)==4):
            #cfR2.append((approx[0]+approx[1]+approx[2]+approx[3])/4)
            M = cv2.moments(approx)
            if M['m00']!=0:
                cx = (M['m10']/M['m00'])
                cy = (M['m01']/M['m00'])
                cfR2.append([cx,cy])
        else:
            M = cv2.moments(approx)
            if M['m00']!=0:
                cx = (M['m10']/M['m00'])
                cy = (M['m01']/M['m00'])
                cfR3.append([cx,cy])

    for c in cfR1:
        ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 1
    for c in cfR2:
        ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 2
    for c in cfR3:
        ar[round(c[1]*10/w)-1,round(c[0]*10/h)-1] = 3
    
    return ar

i=0
flag = True

while i<100:
        
    ###################################### Path Planning

    ar = extract()   

    shapes = {"TY":4, "SY":5, "CY":6, "TR":1, "SR":2, "CR":3}

    n1 = -1
    n2 = -1
    home = -1
    n3 = -1
    
    _,_,start = arucoRead()
    if(i==0):
        grid = start
    if(grid==36):
        n1 = 27
        n2 = 29
        home = 39
        n3 = 38

    elif(grid==44):
        n1 = 53
        n2 = 51
        home = 41
        n3 = 42

    elif(grid==4):
        n1 = 5
        n2 = 23
        home = 31
        n3 = 22

    elif(grid==76):
        n1 = 75
        n2 = 57
        home = 49
        n3 = 58
        
    print('\n',"####################################################################################",'\n')
    
    #print(ar)
    #print(shapes)
    
    val = env.roll_dice()
    print("The value is: ",val)
    ends = []
    minpath = []
    l = 1000
    
        
    for i in range(9):
        for j in range(9):
            if(ar[i][j]==shapes[val]):
                ends.append(i*9+j)

    for node in ends:
        path = bfs(g,start,node)
        if(len(path)>=2 and (len(path)<l or (home in path and len(path)<=l))):
            minpath = path
            l = len(path)
            
    print(start,minpath)
            
    if(home in minpath):
        minpath.append(40)
    for i in range(1,len(minpath),1):
        move(minpath[i])
        print("Arrived at: ",minpath[i])
        time.sleep(0.01)
        if(minpath[i]==40):
            sys.exit("\n\n\nFinished the task!!!\n\n\n")
    if(flag==True):
        if((n1 in minpath) or (n2 in minpath)):
            g.emptyLink(grid,n3)
            g.addEdge(grid,(grid+n3)/2)
            g.addEdge(n3,(grid+n3)/2)
            g.addEdge(n3,home)
            flag = False
        print("Ready for next command")
    
    print('\n',"####################################################################################",'\n')
   
    p.stepSimulation()
       
    #time.sleep(0.05)
    
    i+=1
    
#     if(node==start and len(minpath)>0):
#         i+=1
        
#     else:
#         print("error")
#         continue