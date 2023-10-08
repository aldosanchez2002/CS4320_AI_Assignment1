'''
Pathfinding Assignemnt 1 for CS 4320
Fall 2023

David Dominguez
Lianna Estrada
Aldo Sanchez


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
from pathlib import Path
from copy import deepcopy

class Node:
    def __init__(self, cur_cord, cost_so_far, cur_map, history=[], heuristic=0):
        self.cur_cord = cur_cord
        self.cost_so_far = cost_so_far
        self.cur_map = cur_map
        self.history = history + [str(cur_cord)] #used to store solution path
        self.heuristic = heuristic

    def getNeighbors(self):
        neighbors=[]
        next_map = self.cur_map.copy()
        next_map[self.cur_cord[0]][self.cur_cord[1]] = 0

        for i,j in [(0,1),(0,-1),(1,0),(-1,0)]:  # get coords that are right, left, down and up
            neighborRow = self.cur_cord[0] + i
            neighborColumn = self.cur_cord[1] + j

            if  0 <= neighborRow < len(next_map) and 0 <= neighborColumn < len(next_map[0]):
                mapcost = next_map[neighborRow][neighborColumn]
                if mapcost != 0: #neighbor is passable
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
    This method implements the breadth first search algorithm.
    It is modified to look for the least-costly path instead of the first path to the goal.
    '''
    #nodes_expanded will be derived from the length of the explored nodes set (seen)
    max_nodes_memory=0
    start_time = time.time()

    # used to track the best solution path and cost so far
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
    # break the loop when 3 minutes have passed or there are no more neighbors to visit
    while unvisited_neighbors and time.time() < start_time+180:
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
    if bestSolutionNode and runtime < 180000:
        return bestSolutionCost, nodes_expanded, max_nodes_memory, runtime, bestSolutionNode.getPath()

    # no solution found
    return -1, nodes_expanded, max_nodes_memory, runtime, None

def IterativeDeepeningSearch(map, start, goal):
    '''
    This method implements the iterative deepening search algorithm
    '''
    # nodes_expanded will be derived from the length of the explored nodes set (seen)
    start_time = time.time()
    max_nodes_memory = [0]
    history = []
    # used to track the best solution path and cost so far
    bestSolutionNode = []
    bestSolutionCost = [float('inf')]

    startNode = Node(
        cur_cord=start,
        cost_so_far=map[start[0]][start[1]],
        cur_map=map
    )
    map_copy = deepcopy(map)
    def DepthLimitedSearch(current, goal, limit):
        if current.isSolution(goal): #return end search when goal is found
            history.append(current)
            bestSolutionNode.append(current)
            bestSolutionCost[0] = current.cost_so_far
            return True
        if limit <= 0: # we've reached our depth limit
            return False
        history.append(current)
        unvisited_neighbors = current.getNeighbors()
        max_nodes_memory[0] = max(max_nodes_memory[0], len(unvisited_neighbors))

        if limit <= 5:
            print(f"currentNode={current.cur_cord}")
            print(f"currentMap={current.cur_map}")

        for neighbor in unvisited_neighbors:
            print(f"neighborNode={neighbor.cur_cord}") # used for testing
            # check if any of the neighbors are the goal
            if DepthLimitedSearch(neighbor, goal, limit - 1):
                return True
        return False
    limit = 0
    while time.time() < start_time+180: # while runtime is less than 3 min increase the depth till a solution is found
        if limit <= 5:
            print(f"LIMIT={limit}")
            print(f"OG MAP: \n {map_copy}")
            startNode.cur_map = deepcopy(map_copy) # attempt at a new solution containing original map
        if DepthLimitedSearch(startNode, goal, limit):
            break
        limit += 1
    runtime = (time.time() - start_time) * 1000
    nodes_expanded = len(history)
    if len(bestSolutionNode) and runtime < 180000:
        return bestSolutionCost[0], nodes_expanded, max_nodes_memory, runtime, bestSolutionNode[0].getPath()

    return -1, nodes_expanded, max_nodes_memory, runtime, None




def ManhattanHeuristic(map, current, goal):
    '''
    This method calculates the Manhattan distance heuristic between the current state and the goal
    '''
    pass

def AStarSearch(map, start, goal, heuristic = ManhattanHeuristic):
    '''
    This method implements the A* search algorithm
    '''
    print("Not implemented yet")
    return 0,0,0,0,0

def print_algorithm_output(algorithm_output):
    cost, nodes_expanded, max_nodes, runtime, path = algorithm_output
    print("Cost: " + str(cost))
    print("Nodes Expanded: " + str(nodes_expanded))
    print("Max Nodes: " + str(max_nodes))
    print("Runtime: " + str(runtime))
    print("Path: " + str(path))

def runSearch(start, goal, map, searchType):
    if searchType == 'BFS':
        print_algorithm_output(BreadthFirstSearch(map,start,goal))
    if searchType == 'IDS':
        print_algorithm_output(IterativeDeepeningSearch(map, start, goal))
    if searchType == 'AStar':
        print_algorithm_output(AStarSearch(map,start,goal))

def GenerateTestCase(rows, cols):
    '''
    This method generates Test Cases of size Row x Col with random costs at each coordinate
    '''
    map = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
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
    # searchTypes= ('BFS', 'IDS', 'AStar')
    # if len(sys.argv) != 3:
    #     print("Usage: python Pathfinding.py <map_file> <search-type")
    #     print("Available search types: BFS, IDS, AStar")
    #     sys.exit(1)
    # # generate test cases, run once and comment out
    # if sys.argv[1] == "-G":
    #     testCases = [(5, 5), (10, 10), (15, 15), (20, 20), (100, 100), (5, 10)]
    #     for width, height in testCases:
    #         GenerateTestCase(width, height)
    #     sys.exit(0)
    # target_dir = Path(sys.argv[1])
    #
    # if not target_dir.exists():
    #     print("The target test file doesn't exist. Try again")
    #     raise SystemExit(1)
    # searchType = sys.argv[2]
    # if searchType not in searchTypes:
    #     print("Invalid Search Type. Try again. \n Available search types: BFS, IDS, AStar")
    #     raise SystemExit(1)
    #
    # start, goal, map = readMap(target_dir)
    # runSearch(start,goal,map,searchType)
    '''
    print("*********************** Instructions output ********************")
    start, goal, map = readMap("Testcases/InstructionsMap.txt")

    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))
    '''
    print("*********************** 5x5 output ********************")
    start, goal, map = readMap("Testcases/Map5x5.txt")

    print("\nITERATIVE DEEPENING Search: ")
    print_algorithm_output(IterativeDeepeningSearch(map, start, goal))
    '''
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
    '''

