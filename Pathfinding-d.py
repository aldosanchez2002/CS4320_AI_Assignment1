'''
Pathfinding Assignemnt 1 for CS 4320
Fall 2023
Lianna Estrada
Aldo Sanchez
David Dominguez

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
    def __init__(self, cur_cord, cost_so_far, cur_map, history=[]):

        self.cur_cord = cur_cord
        self.cost_so_far = cost_so_far
        self.cur_map = cur_map
        self.history = history + [str(cur_cord)]

    def getNeighbors(self):
        next_map = self.cur_map.copy()
        next_map[self.cur_cord[0]][self.cur_cord[1]] = 0

        neighbors=[]
        for i,j in [(0,1),(0,-1),(1,0),(-1,0)]:  # right left down up

            neighborRow = self.cur_cord[0] + i
            neighborColumn = self.cur_cord[1] + j

            if  0 <= neighborRow < len(next_map) and 0 <= neighborColumn < len(next_map[0]):

                mapcost = next_map[neighborRow][neighborColumn]
                if mapcost != 0:
                    neighbors.append(Node((neighborRow, neighborColumn), self.cost_so_far+mapcost, next_map, self.history)) # parent is self
        return neighbors
    
    def getPath(self):
        return str(self.history)
    
    def isSolution(self, goal):
        return self.cur_cord[0] == goal[0] and self.cur_cord[1] == goal[1]

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

    
    startNode = Node(
        cur_cord=start,
        cost_so_far=map[start[0]][start[1]],
        cur_map=map
        )
    
    #will hold the unexplored neighbor nodes
    unvisited_neighbors = [startNode]

    history = [] # will hold the explored nodes
    while unvisited_neighbors:
        node = unvisited_neighbors.pop(0)
        history.append(node)
        if node.isSolution(goal):
            if node.cost_so_far < bestSolutionCost:
                bestSolutionNode = node
                bestSolutionCost = node.cost_so_far
            continue

        neighbors = node.getNeighbors()
        for neighbor in neighbors:
            unvisited_neighbors.append(neighbor)
        
        max_nodes_memory = max(max_nodes_memory, len(unvisited_neighbors))

    nodes_expanded = len(history)
    runtime = (time.time() - start_time) * 1000
    if bestSolutionNode:
        return bestSolutionCost, nodes_expanded, max_nodes_memory, runtime, bestSolutionNode.getPath()
    
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

def GenerateTestCase(rows, cols):
    map = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(random.randint(0,5))
        map.append(row)
    start = (random.randint(0,rows-1),random.randint(0,cols-1))
    goal = start
    while start == goal:
        goal = (random.randint(0,rows-1),random.randint(0,cols-1))
    def saveTestCase(start,goal,map, filename):
        with open(filename, "w") as file:
            file.write(str(rows) + " " + str(cols) + "\n")
            file.write(" ".join(str(x) for x in start) + "\n")
            file.write(" ".join(str(x) for x in goal) + "\n")
            for row in map:
                    file.write(" ".join(str(x) for x in row))
                    file.write("\n")
    saveTestCase(start,goal,map,"Map"+str(rows)+"x"+str(cols)+".txt")

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
    print("*********************** 5x5 output ********************")
    start, goal, map = readMap("Testcases/Map5x5.txt")

    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))

    print("*********************** 5x10 output ********************")
    start, goal, map = readMap("Testcases/Map5x10.txt")

    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))

    print("*********************** 10x10 output ********************")

    start, goal, map = readMap("Testcases/Map10x10.txt")

    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))

    print("*********************** 15x15 output ********************")
    start, goal, map = readMap("Testcases/Map15x15.txt")

    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))

    print("*********************** 20x20 output ********************")
    start, goal, map = readMap("Testcases/Map20x20.txt")

    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))

    print("*********************** 100x100 output ********************")
    start, goal, map = readMap("Testcases/Map100x100.txt")

    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))
    sys.exit(0)
    print("\nIterative Deepening Search: ")
    print_algorithm_output(IterativeDeepeningSearch(map, start, goal))

    print("\nA* Search: ")
    print_algorithm_output(AStarSearch(map, start, goal, ManhattanHeuristic))

