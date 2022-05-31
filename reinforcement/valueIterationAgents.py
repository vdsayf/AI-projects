# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        listStates = self.mdp.getStates()


        for k in range(self.iterations):
          vCOPY = self.values.copy()
          for s in listStates:
            if self.mdp.isTerminal(s):
              vCOPY[s] = 0
            else:
              alpha = -float("inf")
              sListActions = self.mdp.getPossibleActions(s)
              for a in sListActions:
                qv = self.computeQValueFromValues(s,a)
                alpha = max(qv, alpha)
              vCOPY[s] = alpha
          self.values = vCOPY

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        summo = 0
        probStateList = self.mdp.getTransitionStatesAndProbs(state,action)
        for ding in probStateList:
          newState = ding[0]
          rew = self.mdp.getReward(state,action,newState)
          summo += ding[1]*(rew + self.discount*(self.getValue(newState)))
        return summo

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        returnThis = None
        alpha = float("-inf")
        for a in self.mdp.getPossibleActions(state):
          qv = self.computeQValueFromValues(state, a)
          if alpha < qv:
            alpha = qv
            returnThis = a
        return returnThis

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Write value iteration code here
        listStates = self.mdp.getStates()


        for k in range(self.iterations):
          vCOPY = self.values.copy()
          s = listStates[k % len(listStates)]
          if self.mdp.isTerminal(s):
            vCOPY[s] = 0
          else:
            alpha = -float("inf")
            sListActions = self.mdp.getPossibleActions(s)
            for a in sListActions:
              qv = self.computeQValueFromValues(s,a)
              alpha = max(qv, alpha)
            vCOPY[s] = alpha

          self.values = vCOPY


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        statelst = self.mdp.getStates()
        preDICT = {}
        prioQ = util.PriorityQueue()


        for st in statelst:
          preDICT[st] = set()

        for st in statelst:
          if not self.mdp.isTerminal(st):
            for a in self.mdp.getPossibleActions(st):
              for pred, prob in self.mdp.getTransitionStatesAndProbs(st,a):
                if prob > 0:
                  preDICT[pred].add(st)

        for st in statelst:
          if not self.mdp.isTerminal(st):
            alpha = float("-inf")
            for a in self.mdp.getPossibleActions(st):
              aqv = self.computeQValueFromValues(st,a)
              if alpha < aqv:
                alpha = aqv
            diff = abs(alpha - self.getValue(st))
            prioQ.push(st,-diff)

        for i in range(self.iterations):
          if prioQ.isEmpty():
            return
          s = prioQ.pop()

          alpha = float("-inf")
          for a in self.mdp.getPossibleActions(s):
            aqv = self.computeQValueFromValues(s,a)
            if alpha < aqv:
              alpha = aqv
          self.values[s] = alpha

          for p in preDICT[s]:
            dingo = float("-inf")
            for a in self.mdp.getPossibleActions(p):
              aqv = self.computeQValueFromValues(p,a)
              if dingo < aqv:
                dingo = aqv
            diff = abs(dingo - self.getValue(p))
            if diff > self.theta:
              prioQ.update(p,-diff)
