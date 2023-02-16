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
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    # To Do: Write all your experiment code here

    # Assignment 1: e-greedy
    env = BanditEnvironment(n_actions=n_actions)
    # pi = EgreedyPolicy(n_actions=n_actions)  # Initialize policy
    # for i in range(n_repetitions):
    #     a = pi.select_action(epsilon=0.1)  # select action
    #     r = env.act(a)  # sample reward
    #     pi.update(a, r)  # update policy
    #     # print("Test e-greedy policy with action {}, received reward {}".format(a, r))

    # Assignment 2: Optimistic init
    # pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    # for i in range(n_repetitions):
    #     a = pi.select_action() # select action
    #     r = env.act(a) # sample reward
    #     pi.update(a,r) # update policy

    plotHelper = LearningCurvePlot(title="UCB average reward over 1000 repetitions")
    vectorResult = np.zeros(n_timesteps)
    for j in range(n_repetitions):
        pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
        for i in range(1, n_timesteps+1):
            a = pi.select_action(c=1.0, t=i)  # select action
            r = env.act(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(a, r)  # update policy
        
    plotHelper.add_curve(vectorResult/float(n_repetitions), "UCB")
    plotHelper.save("UCB.png")
    print("Test UCB policy with action {}, received reward {}".format(a, r))

    # Assignment 3: UCB



    # Assignment 4: Comparison

    pass


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 1000
    n_timesteps = 1000
    smoothing_window = 31

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
