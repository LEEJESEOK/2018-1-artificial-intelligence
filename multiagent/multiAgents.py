# coding=utf-8
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


import random
import util

from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    # 이미 구현되어 있는 코드
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        # 현재 게임 상태에서 음식의 위치 정보를 받아서 저장
        currentFood = currentGameState.getFood()

        # 현재 게임 상태에서 맵의 높이와 너비를 구하기 위해서 벽의 정보를 받아와서 저장
        layout = currentGameState.getWalls()

        # 게임 상에서 팩맨과 유령 혹은 음식이 가장 멀리 있는 경우는 맵의 양 대각선 끝쪽이기 때문에 맵의 높이와 너비를 더한다.=
        maxlength = layout.height - 2 + layout.width - 2

        # 팩맨이 움직이면 올수 있는 다음 게임 상태에 대한 정보를 저장
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # 다음 게임 상태에서 팩맨의 위치 정보를 저장
        newPos = successorGameState.getPacmanPosition()

        # 다음 게임 상태에서 음식의 위치 정보를 저장
        newFood = successorGameState.getFood()

        # 상태를 평가하기 위한 점수를 0 으로 선언
        score = 0

        # 다음 상태에서의 팩맨의 위치가 현재 상태에서의 음식의 위치들중 하나와 맞는지 검사
        if currentFood[newPos[0]][newPos[1]]:
            # 음식을 전부 먹어야지 게임에서 승리하므로 음식의 위치로 가는것에는 점수를 많이 부여한다.
            score += 10

        # 음식과 팩맨의 최소 거리를 구하기 위해서 우선적으로 무한 값으로 선언
        newFoodDistance = float("inf")

        # 다음 상태에서 음식마다 반복문을 실행
        for food in newFood.asList():
            # 음식과 팩맨의 위치간의 거리를 맨하탄 Distance 함수를 이용해서 구한다.
            foodDistance = manhattanDistance(newPos, food)

            # 음식과 팩맨 사이의 위치들중 최소값을 구하기 위해서 min을 사용한다.
            newFoodDistance = min([newFoodDistance, foodDistance])

        # 유령과 팩맨의 최소거리를 구하기 위해서 우선적으로 무한 값으로 선언
        newGhostDistance = float("inf")

        # 다음 상태에서 유령마다 반복문 실행
        for ghost in successorGameState.getGhostPositions():
            # 유령과 팩맨의 위치간의 거리를 맨하탄 Distance 함수를 이용해서 구한다.
            ghostDistance = manhattanDistance(newPos, ghost)

            # 유령과 팩맨 사이의 위치들중 최소값을 구하기 위해서 min을 사용한다.
            newGhostDistance = min([newGhostDistance, ghostDistance])

        # 팩맨과 유령가의 최소 거리가 2보다 작은지 검사
        if newGhostDistance < 2:
            # 유령과 부딫힐 경우 게임에서 패배하기 때문에 팩맨과 유령과의 거리가 2보다 작으면 점수에서 많이 감소시킨다.
            score -= 500

        # 평가하기 위한 점수는 앞에서 구한 음식을 먹었는지 여부에 따른 점수, 유령과 거리가 2보다 작은지의 여부에 따른 점수에
        # 추가적으로 음식과의 최소거리의 역수, 유령과의 최소거리에 거리중 최대로 나올수 있는 maxlength를 나눈 값을 더해준다.
        score = score + 1.0 / newFoodDistance + newGhostDistance / maxlength

        # 점수를 반환한다.
        return score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        def minMaxHelper(gameState, deepness, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness += 1
            # 아래 조건들 중 하나라도 만족하면 평가함수에 현재 상태를 넣고 나온 값을 반환한다.
            if (deepness == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return maxFinder(gameState, deepness, agent)
            else:
                return minFinder(gameState, deepness, agent)

        # 최소 값들 중에서 가장 큰 최대값을 구하는 함수로써 팩맨의 움직임을 결정하는 함수
        def maxFinder(gameState, deepness, agent):
            output = ["meow", -float("inf")]
            # agent가 가능한 action을 리스트로 저장함
            pacActions = gameState.getLegalActions(agent)

            # 가능한 action이 없으면  현재 상태를 저장하고 반환
            if not pacActions:
                return self.evaluationFunction(gameState)

            # 현재 상태에서 가능한 모든 action에 대해 반복
            for action in pacActions:
                # 다음 상태의 평가 값들 중에서 최대값을 구하여 저장한다.
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMaxHelper(currState, deepness, agent + 1)

                if type(currValue) is list:
                    testVal = currValue[1]
                else:
                    testVal = currValue
                # 위에서 구한 값중 최대값을 반환한다.
                if testVal > output[1]:
                    output = [action, testVal]
            return output

        # 유령의 움직임을 결정하는 함수
        def minFinder(gameState, deepness, agent):
            output = ["meow", float("inf")]
            # 유령이 가능한 action을 리스트로 저장함
            ghostActions = gameState.getLegalActions(agent)

            # 가능한 action이 없으면  현재 상태를 저장하고 반환
            if not ghostActions:
                return self.evaluationFunction(gameState)

            # 현재 상태에서 가능한 모든 action에 대해 반복
            for action in ghostActions:
                # 다음 상태의 평가 값들 중에서 최소값을 구하여 저장한다.
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMaxHelper(currState, deepness, agent + 1)
                if type(currValue) is list:
                    testVal = currValue[1]
                else:
                    testVal = currValue
                if testVal < output[1]:
                    output = [action, testVal]
            return output

        outputList = minMaxHelper(gameState, 0, 0)
        return outputList[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # 최소값들 중에서 가장 큰 최대값을 구하는 함수로써 팩맨의 움직임을 결정하는 함수
        def minMaxHelper(gameState, deepness, agent, alpha, beta):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness += 1
            if (deepness == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return maxFinder(gameState, deepness, agent, alpha, beta)
            else:
                return minFinder(gameState, deepness, agent, alpha, beta)

        def maxFinder(gameState, deepness, agent, alpha, beta):
            output = ["meow", -float("inf")]
            pacActions = gameState.getLegalActions(agent)

            if not pacActions:
                return self.evaluationFunction(gameState)

            for action in pacActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMaxHelper(currState, deepness, agent + 1, alpha, beta)

                if type(currValue) is list:
                    testVal = currValue[1]
                else:
                    testVal = currValue

                # real logic
                if testVal > output[1]:
                    output = [action, testVal]
                if testVal > beta:
                    return [action, testVal]
                alpha = max(alpha, testVal)
            return output

        def minFinder(gameState, deepness, agent, alpha, beta):
            output = ["meow", float("inf")]
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            for action in ghostActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = minMaxHelper(currState, deepness, agent + 1, alpha, beta)

                if type(currValue) is list:
                    testVal = currValue[1]
                else:
                    testVal = currValue

                if testVal < output[1]:
                    output = [action, testVal]
                if testVal < alpha:
                    return [action, testVal]
                beta = min(beta, testVal)
            return output

        outputList = minMaxHelper(gameState, 0, 0, -float("inf"), float("inf"))
        return outputList[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expHelper(gameState, deepness, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness += 1
            if (deepness == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return maxFinder(gameState, deepness, agent)
            else:
                return expFinder(gameState, deepness, agent)

        def maxFinder(gameState, deepness, agent):
            output = ["meow", -float("inf")]
            pacActions = gameState.getLegalActions(agent)

            if not pacActions:
                return self.evaluationFunction(gameState)

            for action in pacActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = expHelper(currState, deepness, agent + 1)
                if type(currValue) is list:
                    testVal = currValue[1]
                else:
                    testVal = currValue
                if testVal > output[1]:
                    output = [action, testVal]
            return output

        def expFinder(gameState, deepness, agent):
            output = ["meow", 0]
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            probability = 1.0 / len(ghostActions)

            for action in ghostActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = expHelper(currState, deepness, agent + 1)
                if type(currValue) is list:
                    val = currValue[1]
                else:
                    val = currValue
                output[0] = action
                output[1] += val * probability
            return output

        outputList = expHelper(gameState, 0, 0)
        return outputList[0]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # 현재 게임 상태에서 음식의 위치를 받고 리스트로 저장
    foodPos = currentGameState.getFood().asList()
    # 음식들과 팩맨 사이의 거리들을 저장하기위한 리스트
    foodDist = []
    # 현재 게임에서 팩맨의 위치를 받아서 저장
    currentPos = list(currentGameState.getPacmanPosition())

    # 음식마다 반복문 실행
    for food in foodPos:
        # 팩맨과 음식의 위치간의 거리를 manhattan 함수로 계산
        food2pacmanDist = manhattanDistance(food, currentPos)
"""
        foodDist.append(-1 * food2pacmanDist)

    if not foodDist:
        foodDist.append(0)

    return max(foodDist) + currentGameState.getScore()
"""

# Abbreviation
better = betterEvaluationFunction
