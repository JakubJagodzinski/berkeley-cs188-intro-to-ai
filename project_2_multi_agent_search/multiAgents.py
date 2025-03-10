# multiAgents.py
# --------------
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
from turtledemo.clock import current_day

from graphicsUtils import ghost_shape
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distance_to_nearest_food = float('inf')
        for food_position in newFood.asList():
            distance_to_food = manhattanDistance(newPos, food_position)
            if distance_to_food < distance_to_nearest_food:
                distance_to_nearest_food = distance_to_food

        distance_to_nearest_ghost = float('inf')
        nearest_ghost_index = 0
        for ghost_index, ghost_position in enumerate(successorGameState.getGhostPositions()):
            if newScaredTimes[ghost_index] == 0:
                distance_to_ghost = manhattanDistance(newPos, ghost_position)
                if distance_to_ghost <= 1:
                    return -float('inf')
                if distance_to_ghost < distance_to_nearest_ghost:
                    distance_to_nearest_ghost = distance_to_ghost
                    nearest_ghost_index = ghost_index

        if newPos == currentGameState.getPacmanPosition():
            return -float('inf')

        if newScaredTimes[nearest_ghost_index] == 0:
            if distance_to_nearest_ghost > distance_to_nearest_food:
                return successorGameState.getScore() + (1.0 / distance_to_nearest_food) - (1.0 / distance_to_nearest_ghost)
            else:
                return successorGameState.getScore() - (1.0 / distance_to_nearest_ghost)
        else:
            if distance_to_nearest_ghost > 0:
                return successorGameState.getScore() + (1.0 / distance_to_nearest_food) + (1.0 / distance_to_nearest_ghost)
            else:
                return float('inf')

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def is_not_last_ghost(gameState, agent_nr):
    return agent_nr < (gameState.getNumAgents() - 1)

def is_finished(gameState, current_depth, max_depth):
    return gameState.isWin() or gameState.isLose() or current_depth == max_depth

PACMAN_AGENT_NR = 0
FIRST_GHOST_AGENT_NR = 1
START_DEPTH = 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """



    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        max_value = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(PACMAN_AGENT_NR):
            successor = gameState.generateSuccessor(PACMAN_AGENT_NR, action)
            value = self.get_min(successor, START_DEPTH, FIRST_GHOST_AGENT_NR)
            if value > max_value:
                max_value = value
                best_action = action
        return best_action


    def get_min(self, gameState, current_depth, current_agent_nr):
        if is_finished(gameState, current_depth, self.depth):
            return self.evaluationFunction(gameState)

        min_value = float('inf')
        for current_agent_action in gameState.getLegalActions(current_agent_nr):
            successor = gameState.generateSuccessor(current_agent_nr, current_agent_action)
            if is_not_last_ghost(gameState, current_agent_nr):
                value = self.get_min(successor, current_depth, current_agent_nr + 1)
            else:
                value = self.get_max(successor, current_depth + 1)
            if value < min_value:
                min_value = value

        return min_value


    def get_max(self, gameState, current_depth):
        if is_finished(gameState, current_depth, self.depth):
            return self.evaluationFunction(gameState)

        max_value = float('-inf')
        for current_agent_action in gameState.getLegalActions(PACMAN_AGENT_NR):
            successor = gameState.generateSuccessor(PACMAN_AGENT_NR, current_agent_action)
            value = self.get_min(successor, current_depth, FIRST_GHOST_AGENT_NR)
            if value > max_value:
                max_value = value

        return max_value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        max_value = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(PACMAN_AGENT_NR):
            successor = gameState.generateSuccessor(PACMAN_AGENT_NR, action)
            value = self.get_min(successor, START_DEPTH, FIRST_GHOST_AGENT_NR, max_value, float('inf'))
            if value > max_value:
                max_value = value
                best_action = action
        return best_action

    def get_min(self, gameState, current_depth, current_agent_nr, alpha, beta):
        if is_finished(gameState, current_depth, self.depth):
            return self.evaluationFunction(gameState)

        min_value = float('inf')
        for action in gameState.getLegalActions(current_agent_nr):
            successor = gameState.generateSuccessor(current_agent_nr, action)
            if is_not_last_ghost(gameState, current_agent_nr):
                value = self.get_min(successor, current_depth, current_agent_nr + 1, alpha, beta)
            else:
                value = self.get_max(successor, current_depth + 1, alpha, beta)
            if value < min_value:
                min_value = value
            if min_value < alpha:
                return min_value
            if min_value < beta:
                beta = min_value
        return min_value


    def get_max(self, gameState, current_depth, alpha, beta):
        if is_finished(gameState, current_depth, self.depth):
            return self.evaluationFunction(gameState)

        max_value = float('-inf')
        for action in gameState.getLegalActions(PACMAN_AGENT_NR):
            successor = gameState.generateSuccessor(PACMAN_AGENT_NR, action)
            value = self.get_min(successor, current_depth, FIRST_GHOST_AGENT_NR, alpha, beta)
            if value > max_value:
                max_value = value
            if max_value > beta:
                return max_value
            if max_value > alpha:
                alpha = max_value
        return max_value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        max_value = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(PACMAN_AGENT_NR):
            successor = gameState.generateSuccessor(PACMAN_AGENT_NR, action)
            value = self.get_expected_min(successor, START_DEPTH, FIRST_GHOST_AGENT_NR)
            if value > max_value:
                max_value = value
                best_action = action
        return best_action

    def get_expected_min(self, gameState, current_depth, current_agent_nr):
        if is_finished(gameState, current_depth, self.depth):
            return self.evaluationFunction(gameState)

        values_sum = 0
        actions = gameState.getLegalActions(current_agent_nr)
        for action in actions:
            successor = gameState.generateSuccessor(current_agent_nr, action)
            if is_not_last_ghost(gameState, current_agent_nr):
                values_sum += self.get_expected_min(successor, current_depth, current_agent_nr + 1)
            else:
                values_sum += self.get_max(successor, current_depth + 1)
        return float(values_sum) / len(actions)


    def get_max(self, gameState, current_depth):
        if is_finished(gameState, current_depth, self.depth):
            return self.evaluationFunction(gameState)

        max_value = float('-inf')
        for action in gameState.getLegalActions(PACMAN_AGENT_NR):
            successor = gameState.generateSuccessor(PACMAN_AGENT_NR, action)
            value = self.get_expected_min(successor, current_depth, FIRST_GHOST_AGENT_NR)
            if value > max_value:
                max_value = value
        return max_value


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin():
        return float('inf')
    elif currentGameState.isLose():
        return float('-inf')

    food_positions = currentGameState.getFood().asList()
    capsules_positions = currentGameState.getCapsules()
    pacman_position = currentGameState.getPacmanPosition()
    ghosts_positions = currentGameState.getGhostPositions()
    ghosts_scared_time = [ghost_state.scaredTimer for ghost_state in currentGameState.getGhostStates()]

    ghosts_reward = 0
    ghosts_penalty = 0
    for ghost_index in range(len(ghosts_positions)):
        distance_to_ghost = util.manhattanDistance(pacman_position, ghosts_positions[ghost_index])
        ghost_scared_time = ghosts_scared_time[ghost_index]
        if ghost_scared_time > 0:
            ghosts_reward += (30.0 / (distance_to_ghost + 1))
        else:
            ghosts_penalty += - (10.0 / (distance_to_ghost + 1))

    minimum_distance_to_food = float('inf')
    for food_position in food_positions:
        distance_to_food = util.manhattanDistance(pacman_position, food_position)
        if distance_to_food < minimum_distance_to_food:
            minimum_distance_to_food = distance_to_food
    food_reward = (10.0 / (minimum_distance_to_food + 1))

    minimum_distance_to_capsule = float('inf')
    for capsule_position in capsules_positions:
        distance_to_capsule = util.manhattanDistance(pacman_position, capsule_position)
        if distance_to_capsule < minimum_distance_to_capsule:
            minimum_distance_to_capsule = distance_to_capsule
    capsules_reward = (15.0 / (minimum_distance_to_capsule + 1))

    return currentGameState.getScore() + food_reward + capsules_reward + ghosts_reward + ghosts_penalty

# Abbreviation
better = betterEvaluationFunction
