# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """

    fringe = util.Stack()  # uses a stack object (LIFO)
    fringe.push((problem.getStartState(), []))
    reached_nodes = {problem.getStartState()}
    # fringe keeps tuples of length 2 containing the node position tuple and
    # a list containing the actions taken to get to that node.
    # an unordered set is used to store all the nodes already reached, and is
    # initialized with the starting node already in the set.

    while not fringe.isEmpty():
        node, path = fringe.pop()

        if problem.isGoalState(node):
            return path
        else:
            for successor, action, stepcost in problem.getSuccessors(node):
                if successor not in reached_nodes:
                    reached_nodes.add(node)
                    new_path = path + [action]
                    fringe.push((successor, new_path))


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    fringe = util.Queue()  # uses a queue object (FIFO)
    fringe.push((problem.getStartState(), []))
    reached_nodes = {problem.getStartState()}
    # fringe keeps tuples of length 2 containing the node position tuple and
    # a list containing the actions taken to get to that node.
    # an unordered set is used to store all the nodes already reached, and is
    # initialized with the starting node already in the set.

    while not fringe.isEmpty():
        node, path = fringe.pop()

        if problem.isGoalState(node):
            return path
        else:
            for successor, action, stepcost in problem.getSuccessors(node):
                if successor not in reached_nodes:
                    reached_nodes.add(successor)
                    new_path = path + [action]
                    fringe.push((successor, new_path))


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    fringe = util.PriorityQueue()  # uses priority queue object
    fringe.push((problem.getStartState(), [], 0), 0)
    reached_nodes = {problem.getStartState(): 0}
    # fringe keeps tuples of length 3 containing the node position tuple, a
    # list containing the actions taken to get to that node, and the total cost
    # of taking all those actions.
    # a dictionary is used to store all the nodes already reached, with the
    # values being the cost of reaching that node. It is initialized with the
    # starting node with a cost of 0.

    while not fringe.isEmpty():
        node, path, total_cost = fringe.pop()

        if problem.isGoalState(node):
            return path
        else:
            for successor, action, stepcost in problem.getSuccessors(node):
                new_path = path + [action]
                new_cost = total_cost + stepcost
                if successor not in reached_nodes.keys() \
                        or new_cost < reached_nodes[successor]:
                    reached_nodes[successor] = new_cost
                    fringe.push((successor, new_path, new_cost),
                                new_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    fringe = util.PriorityQueue()  # uses a priority queue object
    fringe.push((problem.getStartState(), [], 0), 0)
    reached_nodes = {problem.getStartState(): heuristic(problem.getStartState(),
                                                        problem)}
    # fringe keeps tuples of length 3 containing the node position tuple, a
    # list containing the actions taken to get to that node, and the total cost
    # of taking all those actions + the heuristic value.
    # a dictionary is used to store all the nodes already reached, with the
    # values being the cost of reaching that node. It is initialized with the
    # starting node with a cost of the heuristic of the starting node.

    while not fringe.isEmpty():
        node, path, total_cost = fringe.pop()

        if problem.isGoalState(node):
            return path
        else:
            total_cost -= heuristic(node, problem)  # remove previous heuristic
            for successor, action, stepcost in problem.getSuccessors(node):
                new_path = path + [action]
                new_cost = total_cost + stepcost + heuristic(successor,
                                                             problem)
                if successor not in reached_nodes.keys() \
                        or new_cost < reached_nodes[successor]:
                    reached_nodes[successor] = new_cost
                    fringe.update((successor, new_path, new_cost),
                                  new_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
