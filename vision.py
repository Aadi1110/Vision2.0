"""
Importing necessary libraries 

cv2.aruco -> detecting aruco marker
vision_arena -> custom environment
pybullet -> physics engine for simulation
"""
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

class VisionENV():
    
    def __init__(self):
        """
        Constructor function for building the Vision Environment
        Argumets:
            None
        Return:
            None
        """
        self.env = gym.make("vision_arena-v0")
        self.img = self.env.camera_feed(is_flat=True)
        
        self.width, self.height = int(self.img.shape[1]), int(self.img.shape[0])
        """
        s1 -> maximum value of rotational speed
        s2 -> maximum value of translational speed
        ka, kt -> PID parameters
        """
        self.s1, self.s2 = 250, 350
        self.ka, self.kt = 0.8, 1.0

    def arucoRead(self):
        """
        Argumets:
            None
        Returns:
            (Center of Bot, Grid Number of Bot, Vector for Bot's movement)
        """
        
        img = self.env.camera_feed(is_flat=True)
        p.stepSimulation()
        
        ARUCO_PARAMETERS = aruco.DetectorParameters_create() # constant parameters used in aruco methods
        ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        
        board = aruco.GridBoard_create(markersX=2, markersY=2, markerLength=0.09, markerSeparation=0.01, dictionary=ARUCO_DICT)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert arena image to grayscale

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS) # detect aruco markers

        center = np.array((corners[0][0][0]+corners[0][0][1]+corners[0][0][2]+corners[0][0][3])/4)
        grid = round(center[1]*10/self.height-1)*9+round(center[0]*10/self.width-1)
        bvec = np.array([(corners[0][0][0][0]-corners[0][0][3][0]),(corners[0][0][0][1]-corners[0][0][3][1])])

        return(center, bvec, grid)

    def dist(self,p1,p2):
        """
        Argument:
            Vectors p1 and p2
        Returns:
            Distance between vectors p1 and p2
        """
        return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    def ang(self,v,bvec):
        """
        Argument:
            Vectors v and bvec
        Returns:
            Angle between vectors v and bvec
        """
        c1 = complex(bvec[0],bvec[1])
        c2 = complex(v[0],v[1])
        
        angle = np.angle(c1/c2)
        angle = math.degrees(angle)
        
        return angle

    def rotate(self,tovec):
        """
        Aligns the vector fo the bot with the destination vector
        Argument:
            Destination Vector
        Returns:
            None
        """
        while True:
            
            cenBot, bvec, pos = self.arucoRead() # get parameters of the bot
            
            v = tovec - cenBot
            angle = self.ang(v,bvec)
            
            sr = int(self.s1 - self.ka*(90-abs(angle)))
            
            if(self.dist(tovec,cenBot)>15):
                
                if(abs(angle)<=10):
                    break
                    
                if(angle>0):
                    
                    if(angle>90):
                        
                        for i in range(sr):
                            
                            p.stepSimulation()
                            self.env.move_husky(-1,1,-1,1) # move left
        
                    else:
                        
                        for i in range(sr):
                            
                            p.stepSimulation()
                            self.env.move_husky(-0.6,1,-0.6,1) # move left
                else:
                    
                    if(angle<-90):
                        
                        for i in range(sr):
                            
                            p.stepSimulation()
                            self.env.move_husky(1,-1,1,-1) # move right
        
                    else:
                        
                        for i in range(sr):
                            
                            p.stepSimulation()
                            self.env.move_husky(1,-0.6,1,-0.6) # move right
            else:
                
                for i in range(self.s1):
                    
                    p.stepSimulation()
                    self.env.move_husky(0,0,0,0) # stop
                    
                break

    def move(self,toGrid):
        """
        Function to move the bot to given Grid Number
        Argument:
            Destination Grid Number
        Returns:
            None
        """
        tovec = np.array([(toGrid%9+1)*self.height/10, (toGrid//9+1)*self.width/10])
        
        if(toGrid>=0 and toGrid<9):
            tovec[1] = tovec[1]-7
            
        if(toGrid%9==0):
            tovec[0] = tovec[0]-7
            
        img = self.env.camera_feed(is_flat=True)
        f=1
        
        while True:
            
            cenBot, bvec,pos = self.arucoRead() # get parameters of the bot
            
            d = self.dist(tovec,cenBot)
            st = int(self.s2 - self.kt*(60-d))
            
            if(d<15):
                break
                
            self.rotate(tovec)
            
            for i in range(st):
                
                p.stepSimulation()
                self.env.move_husky(1,1,1,1) # move forward

    class Graph:
        
        def __init__(self):
            """
            Constructor function for initializing our Graph
            Argument:
                None
            Returns:
                None
            """
            self.graph = collections.defaultdict(list)
            
        def addEdge(self,u,v):
            """
            Add edge between vertices u and v
            Argument:
                Vertices u and v
            Returns:
                None
            """
            self.graph[u].append(v)
            
        def getNeighbors(self,node):
            """
            Argument:
                Vertex of graph
            Returns:
                Neighbours of given node
            """
            return self.graph[node]
        
        def emptyLink(self,n1,n2):
            """
            Disconnect vertices n1 and n2 from graph
            Argument:
                Vertices n1 and n2
            Returns:
                None
            """
            self.graph[n1] = []
            self.graph[n2] = []

        def bfs(self,graph, start, goal):
            """
            Employ BFS (Breadth First Search) to compute path from one vertex to another in the graph
            Argument:
                (Graph, Starting Vertex, Destination Vertex)
            Returns:
                Path from Starting vertex to Destination Vertex
            """
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
        
    def extract(self):
        """
        Extract shapes of different colors using segmentation
        Argument:
            None
        Returns:
            Array describing all the shapes present in the arena
            {
            1 -> Red Triangle
            2 -> Red Square
            3 -> Red Circle
            4 -> Yellow Triangle
            5 -> Yellow Square
            6 -> Yellow Circle
            }
        """
        lY = np.array([20,100,100]) # range for yellow color for masking
        uY = np.array([40,255,255])
        
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV) # convert arena image (BGR) to HSV
        
        maskY = cv2.inRange(hsv,lY,uY)
        kernel = np.ones((1,1), np.uint8)
        
        erY = cv2.erode(maskY, kernel, iterations = 1) # erosion and dilation 
        dilY = cv2.dilate(erY, kernel, iterations = 1)

        contoursY,_ = cv2.findContours(dilY, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        
        cfY1=[] # centers of yellow traingles
        cfY2=[] # centers of yellow squares
        cfY3=[] # centers of yellow circles

        for cnt in contoursY:
            
            eps = 0.02*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            
            if(len(approx)==3):
                
                M = cv2.moments(approx)
                
                if M['m00']!=0:
                    
                    cx = (M['m10']/M['m00'])
                    cy = (M['m01']/M['m00'])
                    cfY1.append([cx,cy])
                    
            elif(len(approx)==4):
                
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
            ar[round(c[1]*10/self.width)-1,round(c[0]*10/self.height)-1] = 4
        for c in cfY2:
            ar[round(c[1]*10/self.width)-1,round(c[0]*10/self.height)-1] = 5
        for c in cfY3:
            ar[round(c[1]*10/self.width)-1,round(c[0]*10/self.height)-1] = 6

        lR = np.array([0,100,100]) # range for red color for masking
        uR = np.array([10,255,255])
        
        maskR = cv2.inRange(hsv,lR,uR)
        
        erR = cv2.erode(maskR, kernel, iterations = 1) # erosion and dilation
        dilR = cv2.dilate(erR, kernel, iterations = 1)
        
        contoursR,_ = cv2.findContours(dilR, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cfR1=[] # centers of red traingles
        cfR2=[] # centers of red squares
        cfR3=[] # centers of red circles

        for cnt in contoursR:
            
            eps = 0.02*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            
            if(len(approx)==3):
                
                M = cv2.moments(approx)
                
                if M['m00']!=0:
                    
                    cx = (M['m10']/M['m00'])
                    cy = (M['m01']/M['m00'])
                    cfR1.append([cx,cy])
                    
            elif(len(approx)==4):
               
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
            ar[round(c[1]*10/self.width)-1,round(c[0]*10/self.height)-1] = 1
        for c in cfR2:
            ar[round(c[1]*10/self.width)-1,round(c[0]*10/self.height)-1] = 2
        for c in cfR3:
            ar[round(c[1]*10/self.width)-1,round(c[0]*10/self.height)-1] = 3

        return ar
        
bot = VisionENV()

g = bot.Graph()
"""
"""
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
g.addEdge(38,37)
g.addEdge(58,67)
g.addEdge(42,43)
g.addEdge(22,13)
g.addEdge(13,4)
g.addEdge(13,22)
g.addEdge(37,38)
g.addEdge(37,36)
g.addEdge(43,44)
g.addEdge(43,42)
g.addEdge(67,76)
g.addEdge(67,58)

i=0
flag = True

ar = np.zeros((9,9), dtype=int)

"""
"""
while i<100:

    ar = bot.extract()   

    shapes = {"TY":4, "SY":5, "CY":6, "TR":1, "SR":2, "CR":3}

    n1 = -1
    n2 = -1
    home = -1
    n3 = -1
    
    _,_,start = bot.arucoRead()
    
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
    
    val = bot.env.roll_dice()
    print("The value is: ",val)
    ends = []
    minpath = []
    l = 1000
    
        
    for i in range(9):
        for j in range(9):
            
            if(ar[i][j]==shapes[val]):
                ends.append(i*9+j)

    for node in ends:
        
        path = g.bfs(g,start,node)
        
        if(len(path)>=2 and (len(path)<l or (home in path and len(path)<=l))):
            minpath = path
            l = len(path)
            
    print(start,minpath)
            
    if(home in minpath):
        
        minpath.append(40)
        
    for i in range(1,len(minpath),1):
        
        bot.move(minpath[i])
        print("Arrived at: ",minpath[i])
        time.sleep(0.01)
        
        if(minpath[i]==40):
            sys.exit("Finish")
            
    if(flag==True):
        
        if((n1 in minpath) or (n2 in minpath)):
            
            g.emptyLink(grid,n3)
            g.addEdge(grid,(grid+n3)/2)
            g.addEdge(n3,(grid+n3)/2)
            g.addEdge(n3,home)
            flag = False
   
    p.stepSimulation()
    
    i+=1
