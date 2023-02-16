#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment


class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # TO DO: Add own code
        self.estimates = np.zeros(n_actions)
        self.steps = np.zeros(n_actions)
        pass

    def select_action(self, epsilon):
        # TO DO: Add own code
        # Replace this with correct action selection
        # generate a random number between 0 and 1 and store it in rand using numpy
        # if rand is smaller than epsilon, select a random action
        # otherwise, select the greedy action
        rand = np.random.uniform(0, 1)
        if rand < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.estimates)
            print("Greedy: " + str(self.estimates) + " with action " + str(a))

        return a

    def update(self, a, r):
        # TO DO: Add own code
        self.steps[a] += 1
        self.estimates[a] += (1/self.steps[a])*(r-self.estimates[a])
        pass


class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.estimates = np.zeros(n_actions)
        self.learning_rate = learning_rate
        pass

    def select_action(self):
        # TO DO: Add own code
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return np.argmax(self.estimates)

    def update(self, a, r):
        # TO DO: Add own code
        self.estimates[a] += self.learning_rate * (r-self.estimates[a])
        pass


class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.estimates = np.zeros(n_actions)
        self.steps = np.zeros(n_actions)
        self.steps += 1e-3
        # TO DO: Add own code
        pass

    def select_action(self, c, t):
        # make sure we do not divide by zero
        x = np.argmax(self.estimates + c * np.sqrt(np.log(t) / (self.steps)))
        return x

    def update(self, a, r):
        self.steps[a] += 1
        self.estimates[a] += (1/self.steps[a])*(r-self.estimates[a])
        pass

class GradientBanditAlgorithm:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.totalReward = 0
        self.preferences = np.zeros(n_actions)
        self.steps = 1
        pass

    def policy(self):
        return np.exp(self.preferences) / np.sum(np.exp(self.preferences))

    def select_action(self):
        # make sure we do not divide by zero
        return np.argmax(self.policy())

    def update(self, a, r, alpha):
        self.totalReward += r
        self.preferences[a] += alpha * (r - self.totalReward / self.steps) * (1 - self.policy()[a])
        for (i, p) in enumerate(self.preferences):
            if i != a:
                self.preferences[i] -= alpha * (r - self.totalReward / self.steps) * self.policy()[i]
        self.steps += 1
        pass


def test():

    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions)  # Initialize environment

    pi = EgreedyPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(epsilon=0.5)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a, r))

    pi = OIPolicy(n_actions=n_actions, initial_value=1.0)  # Initialize policy
    a = pi.select_action()  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a, r))

    pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(c=1.0, t=1)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test UCB policy with action {}, received reward {}".format(a, r))

    pi = GradientBanditAlgorithm(n_actions=n_actions)  # Initialize policy
    a = pi.select_action()  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r, 0.1)  # update policy
    print("Test gradientBandit policy with action {}, received reward {}".format(a, r))


if __name__ == '__main__':
    test()
