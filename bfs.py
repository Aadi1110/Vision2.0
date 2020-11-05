import collections
class Graph:
	def __init__(self):
		self.graph = collections.defaultdict(list)
	def addEdge(self,u,v):
		self.graph[u].append(v)
	def getNeighbors(self,node):
		return self.graph[node]

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
g.addEdge(38,39)
g.addEdge(38,37)
g.addEdge(58,67)
g.addEdge(58,49)
g.addEdge(42,43)
g.addEdge(42,41)
g.addEdge(22,31)
g.addEdge(22,13)
g.addEdge(13,4)
g.addEdge(13,22)
g.addEdge(37,38)
g.addEdge(37,36)
g.addEdge(43,44)
g.addEdge(43,42)
g.addEdge(67,76)
g.addEdge(67,58)

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

print(bfs(g,38,39))
