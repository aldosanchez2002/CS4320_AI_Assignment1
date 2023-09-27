'''
Pathfinding Assignemnt 1 for CS 4320
Fall 2023
Lianna Estrada
Aldo Sanchez
David LASTNAME

Each pathfinding algorithm should return a tuple containing the following:
    1) Cost of the path
    2) Amount Nodes Expanded
    3) Maximum number of nodes held in memory
    4) Runtime of the algorithm in milliseconds
    5) Path coordinates
'''


import sys
import time
import random

class Node:
    def __init__(self, parent, row, column, cost):
        self.parent = parent
        self.row = row 
        self.column = column
        self.cost = cost
        self.heuristic = 0

    def getNeighbors(self,map):
        neighbors=[]
        for i,j in [(0,1),(0,-1),(1,0),(-1,0)]:  # right left down up
            neighborRow = self.row + i
            neighborColumn = self.column + j
            if  0 <= neighborRow < len(map) and 0 <= neighborColumn < len(map[0]):
                mapcost = map[neighborRow][neighborColumn]
                if mapcost != 0:
                    neighbors.append(Node(self,neighborRow,neighborColumn,self.cost+mapcost)) # parent is self
        return neighbors
    
    def getPath(self, start):
        curCoord = ["("+str(self.row) + "," + str(self.column)+")"]
        if not self.parent:
            return curCoord
        return self.parent.getPath(start) + curCoord
    
    def onCoordinate(self, row, column):
        return self.row == row and self.column == column
    
    def __eq__(self, other):
        return self.row == other.row and self.cost == other.cost and self.column == other.column
    
    def __hash__(self) -> int:
        return hash((self.row, self.column, self.cost))

def readMap(filename):
    '''
    This method reads the map from a file and returns the start coordinates, goal coordinates, and the map
    '''
    with open(filename, "r") as file:
        _ = file.readline()
        start = tuple(int(x) for x in file.readline().split())
        goal = tuple(int(x) for x in file.readline().split())
        map = []
        for line in file:
            map.append([int(x) for x in line.split()])
        return start, goal, map

def BreadthFirstSearch(map, start, goal, seen = set()):
    '''
    This method implements the breadth first search algorithm
    '''
    cost=0
    #nodes_expanded will be derived from the length of the explored nodes set (seen)
    max_nodes_memory=0
    start_time = time.time()

    bestSolutionNode = None
    bestSolutionCost = float('inf')

    #neighbors will hold the unexplored neighbor nodes
    startNode = Node(
        parent = None,
        row = start[0],
        column = start[1],
        cost = map[start[0]][start[1]]
        )
    seen = set()    #seen will hold the explored nodes
    unvisited_neighbors = [startNode]

    while unvisited_neighbors:
        node = unvisited_neighbors.pop(0)
        seen.add(node)                 
        # print("coordinate", node.row, node.column, "Running cost", node.cost, "Map Cost", map[node.row][node.column])
        if node.onCoordinate(goal[0], goal[1]):
            if node.cost < bestSolutionCost:
                bestSolutionNode = node
                bestSolutionCost = node.cost
        neighbors = node.getNeighbors(map)
        for neighbor in neighbors:
            if neighbor not in seen and neighbor.cost < bestSolutionCost:
                unvisited_neighbors.append(neighbor)
                neighbor
        max_nodes_memory = max(max_nodes_memory, len(unvisited_neighbors) + len(seen))

    nodes_expanded = len(seen)
    runtime = (time.time() - start_time) * 1000
    if bestSolutionNode:
        return bestSolutionCost, nodes_expanded, max_nodes_memory, runtime, bestSolutionNode.getPath(start)
    
    # no solution found - should not run
    return -1, nodes_expanded, max_nodes_memory, runtime, None

def IterativeDeepeningSearch(map, start, goal):
    '''
    This method implements the iterative deepening search algorithm
    '''
    pass

def ManhattanHeuristic(map, current, goal):
    '''
    This method calculates the Manhattan distance heuristic between the current state and the goal
    '''
    pass

def AStarSearch(map, start, goal, heuristic = ManhattanHeuristic):
    '''
    This method implements the A* search algorithm
    '''
    pass

def print_algorithm_output(algorithm_output):
    cost, nodes_expanded, max_nodes, runtime, path = algorithm_output
    print("Cost: " + str(cost))
    print("Nodes Expanded: " + str(nodes_expanded))
    print("Max Nodes: " + str(max_nodes))
    print("Runtime: " + str(runtime))
    print("Path: " + str(path))

def GenerateTestCase(width,height):
    map = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(random.randint(0,5))
        map.append(row)
    start = (random.randint(0,height-1),random.randint(0,width-1))
    goal = start
    while start == goal:
        goal = (random.randint(0,height-1),random.randint(0,width-1))
    def saveTestCase(start,goal,map, filename):
        with open(filename, "w") as file:
            file.write("".join(str(x) for x in start) + "\n")
            file.write("".join(str(x) for x in goal) + "\n")
            for row in map:
                    file.write(" ".join(str(x) for x in row))
                    file.write("\n")
    saveTestCase(start,goal,map,"Map"+str(width)+"x"+str(height)+".txt")

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: python Pathfinding.py <map_file>")
    #     sys.exit(1)
    # #generate test cases, run once and comment out
    # if sys.argv[1] == "-G":
    #     testCases = [(5,5), (10,10), (15,15),(20,20), (100,100), (5,10)]
    #     for width,height in testCases:
    #         GenerateTestCase(width,height)
    #     sys.exit(0)

    #Run Algorithms
    # start, goal, map = readMap(sys.argv[1])
    start, goal, map = readMap("Testcases/Map3x3.txt")

    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))
    sys.exit(0)
    print("\nIterative Deepening Search: ")
    print_algorithm_output(IterativeDeepeningSearch(map, start, goal))

    print("\nA* Search: ")
    print_algorithm_output(AStarSearch(map, start, goal, ManhattanHeuristic))

