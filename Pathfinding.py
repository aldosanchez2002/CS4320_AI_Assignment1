import sys

def readMap(filename):
    '''
    This method reads the map from a file and returns the start coordinates, goal coordinates, and the map
    '''
    pass

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

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python Pathfinding.py <map_file>")
        sys.exit(1)
    start, goal, map = readMap(sys.argv[1])
    print("\nBreadth First Search: ")
    print_algorithm_output(BreadthFirstSearch(map, start, goal))
    print("\nIterative Deepening Search: ")
    print_algorithm_output(IterativeDeepeningSearch(map, start, goal))
    print("\nA* Search: ")
    print_algorithm_output(AStarSearch(map, start, goal, ManhattanHeuristic))
    
