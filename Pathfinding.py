import sys
import time
import random

def readMap(filename):
    '''
    This method reads the map from a file and returns the start coordinates, goal coordinates, and the map
    '''
    with open(filename, "r") as file:
        start = tuple(int(x) for x in file.readline().split())
        goal = tuple(int(x) for x in file.readline().split())
        map = []
        for line in file:
            map.append([int(x) for x in line.split()])
        return start, goal, map

def BreadthFirstSearch(map, start, goal):
    '''
    This method implements the breadth first search algorithm
    '''
    pass

def IterativeDeepeningSearch(map, start, goal):
    '''
    This method implements the iterative deepening search algorithm
    '''
    pass

def AStarSearch(map, start, goal, heuristic):
    '''
    This method implements the A* search algorithm
    '''
    pass

def ManhattanHeuristic(map, current, goal):
    '''
    This method calculates the Manhattan distance heuristic between the current state and the goal
    '''
    pass

'''
Each pathfinding algorithm should return a tuple containing the following:
    1) Cost of the path
    2) Amount Nodes Expanded
    3) Maximum number of nodes held in memory
    4) Runtime of the algorithm in milliseconds
    5) Path coordinates
'''

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
    if len(sys.argv) != 2:
        print("Usage: python Pathfinding.py <map_file>")
        sys.exit(1)
    testCases = [(5,5), (10,10), (15,15),(20,20), (100,100), (5,10)]
    for width,height in testCases:
        GenerateTestCase(width,height)
    # start, goal, map = readMap(sys.argv[1])
    # print("\nBreadth First Search: ")
    # print_algorithm_output(BreadthFirstSearch(map, start, goal))
    # print("\nIterative Deepening Search: ")
    # print_algorithm_output(IterativeDeepeningSearch(map, start, goal))
    # print("\nA* Search: ")
    # print_algorithm_output(AStarSearch(map, start, goal, ManhattanHeuristic))

