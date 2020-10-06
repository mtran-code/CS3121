# searchAgents.py
# ---------------
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


from game import Directions
from game import Agent
from game import Actions
import util
import time
import search


class GoWestAgent(Agent):
    """
    An agent that goes West until it can't.
    """

    def getAction(self, state):
        """
        The agent receives a GameState (defined in pacman.py).
        """
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem',
                 heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(
                fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(
                    heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (
                fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(
                prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction is None:
            raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.searchFunction(problem)  # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (
            totalCost, time.time() - starttime))
        if '_expanded' in dir(problem):
            print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self):
            self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None,
                 warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start is not None:
            self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (
                gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(
                        __main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(
                        self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST,
                       Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions is None:
            return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.costFn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn,
                                                              (1, 1), None,
                                                              False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    """
    The Manhattan distance heuristic for a PositionSearchProblem
    """
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    """
    The Euclidean distance heuristic for a PositionSearchProblem
    """
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0

        self.reached_corners = (False, False, False, False)
        # Adds a tuple of four bool values corresponding to each corner of the
        # problem (bottom left, top left, bottom right, top right) and whether
        # or not they have been previously reached. Initializes as all False.

    def getStartState(self):
        """
        Returns the start state.
        """
        # adds tuple of reached_corners to node.
        return self.startingPosition, self.reached_corners

    def isGoalState(self, state):
        """
        Returns whether search state is a goal state of the problem.
        """
        position, reached_corners = state
        for corner in reached_corners:
            if not corner:
                return False
        # check reached_corners tuple to see if any have yet to be reached. If
        # there is any value False (i.e. not reached yet), return False (i.e.
        # not goal state).
        return True

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """
        successors = []
        position, reached_corners = state
        for action in [Directions.NORTH,
                       Directions.SOUTH,
                       Directions.EAST,
                       Directions.WEST]:
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if not self.walls[nextx][nexty]:
                temp = list(reached_corners)
                for i, corner in enumerate(self.corners):
                    if (nextx, nexty) == corner:
                        temp[i] = True
                # change reached_corners tuple into a list temporarily in order
                # to update its values. If the current position is a corner,
                # update its reached status in the list as True. Add the tuple
                # back into the node after.

                next_state = ((nextx, nexty), tuple(temp))  # back to tuple
                next_action = action
                next_cost = 1
                successors.append((next_state,
                                   next_action,
                                   next_cost))

        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.
        """
        if actions is None:
            return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    Function is admissible and consistent.
    """
    heuristic = 0  # initialize heuristic variable as zero
    position, corners_reached = state
    corners = problem.corners
    walls = problem.walls
    maze_width = walls.width - 3
    maze_height = walls.height - 3
    # 3 is subtracted rom the maze width and height to account for the border
    # and the overlap with the current position (i.e. width = the distance from
    # left to right sides of the maze if there were no walls in the way), etc.

    corners_remaining = 0
    for corner in corners_reached:
        if not corner:
            corners_remaining += 1
    # count how many corners haven't been reached yet (0 - 4)

    if corners_remaining == 4:
        if maze_width >= maze_height:
            heuristic += maze_width + 2 * maze_height
        elif maze_width < maze_height:
            heuristic += 2 * maze_width + maze_height
    # if there are still four corners remaining, pacman still has to travel at
    # least the distance between all four of them, i.e. span the width and
    # height of the maze at least one, as well as one more span (either the
    # width or height, whichever is less to ensure admissibility).

    elif corners_remaining == 3:
        heuristic += maze_width + maze_height
    # if there are three corners remaining, pacman still has to travel at least
    # the height and width of the maze once, from a top corner to bottom corner,
    # and from a left corner to right corner.

    elif corners_remaining == 2:
        # **REFERENCE: Corners are (BOTTOMLEFT, TOPLEFT, BOTTOMRIGHT, TOPRIGHT)
        # if there are two corners remaining, there are multiple cases to take
        # into consideration:

        if corners_reached == (False, False, True, True) \
                or corners_reached == (True, True, False, False):
            heuristic += maze_height
        # if the two corners are on the bottom and top of the same side of the
        # maze, then pacman must travel at least the height of the maze to
        # reach them both (the distance between the two corners).

        elif corners_reached == (False, True, False, True) \
                or corners_reached == (True, False, True, False):
            heuristic += maze_width
        # if the two corners are on the left and right of the same side of the
        # maze, then pacman must travel at least the width of the maze to reach
        # them both (the distance between the two corners).

        elif corners_reached == (False, True, True, False) \
                or corners_reached == (True, False, False, True):
            heuristic += abs(corners[0][0] - corners[3][0]) \
                         + abs(corners[0][1] - corners[3][1])
        # if the two corners are diagonal to one another, pacman must travel at
        # least the manhattan distance of the diagonal to reach them both (the
        # distance between the two corners).

    # in addition to the distance between remaining corners, the distance to
    # the nearest remaining corner to pacman can be added to heuristic cost
    # function.
    manhattan_distances = list()
    for i, corner in enumerate(corners):
        if not corners_reached[i]:
            manhattan_distances.append(abs(position[0] - corner[0])
                                       + abs(position[1] - corner[1]))
    if manhattan_distances:  # ensure manhattan distances list is not empty
        heuristic += min(manhattan_distances)
    # go through the corners yet to be reached, and calculate the manhattan
    # distance from current position. From all these values, get the
    # lowest value (i.e. closest corner) and add that to heuristic.

    # summary: this heuristic returns the manhattan distance to the nearest
    # corner added to the manhattan distance between all remaining corners.
    return heuristic


class AStarCornersAgent(SearchAgent):
    """
    A SearchAgent for FoodSearchProblem using A* and your foodHeuristic
    """

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob,
                                                              cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False,
                      specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(),
                      startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {}

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
        """
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH,
                          Directions.SOUTH,
                          Directions.EAST,
                          Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    """
    A SearchAgent for FoodSearchProblem using A* and your foodHeuristic
    """

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob,
                                                              foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    heuristic = 0
    position, foodGrid = state
    foods = foodGrid.asList()

    if foods:
        bottommost = (1, 999999)
        topmost = (1, 1)
        leftmost = (999999, 1)
        rightmost = (1, 1)
        for food_x, food_y in foods:
            if food_y < bottommost[1]:
                bottommost = (food_x, food_y)
            if food_y > topmost[1]:
                topmost = (food_x, food_y)
            if food_x < leftmost[0]:
                leftmost = (food_x, food_y)
            if food_x > rightmost[0]:
                rightmost = (food_x, food_y)

        heuristic += (topmost[1] - bottommost[1])
        heuristic += (rightmost[0] - leftmost[0])

    manhattan_distances = list()
    for food in foods:
        manhattan_distances.append(abs(position[0] - food[0])
                                   + abs(position[1] - food[1]))
    if manhattan_distances:  # ensure manhattan distances list is not empty
        heuristic += min(manhattan_distances)

    return heuristic


class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches
    """

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while currentState.getFood().count() > 0:
            nextPathSegment = self.findPathToClosestDot(currentState)
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception(
                        'findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        util.raiseNotDefined()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        """
        Stores information from the gameState.  You don't need to change this.
        """
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state

        if (x, y) in self.food:
            return True
        else:
            return False


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2,
                                 warn=False, visualize=False)
    return len(search.bfs(prob))
