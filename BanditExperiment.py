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

    # env = BanditEnvironment(n_actions=n_actions)
    # Assignment 1: egreedy
    eHyper = [0.01, 0.05, 0.1, 0.25]
    plotHelper = LearningCurvePlot(
        title="Average performance e-greedy over {} repetitions".format(n_repetitions))

    for e in eHyper:
        print("running e-greedy value "+ str(e))
        result = run_egreedy(n_actions, n_timesteps, n_repetitions, e)
        plotHelper.add_curve(
            smooth(result/float(n_repetitions), window=smoothing_window), label='e-greedy value '+ str(e))

    plotHelper.save("egreedy.png")
    # Assignment 4: Comparison
    # run_compare(n_actions, n_timesteps, n_repetitions, smoothing_window)

    pass


def run_egreedy(n_actions, n_timesteps, n_repetitions, eHyper):
    vectorResult = np.zeros(n_timesteps)
    for j in range(n_repetitions):
        # Initialize policy
        env = BanditEnvironment(n_actions=n_actions)
        pi = EgreedyPolicy(n_actions=n_actions)

        for i in range(1, n_timesteps+1):
            a = pi.select_action(epsilon=eHyper)  # select action
            r = env.act(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(a, r)  # update policy

    return vectorResult


def run_IO(n_actions, n_timesteps, n_repetitions, learnHyper, initHyper):
    vectorResult = np.zeros(n_timesteps)
    for j in range(n_repetitions):
        # Initialize policy
        env = BanditEnvironment(n_actions=n_actions)
        pi = OIPolicy(n_actions=n_actions,
                      initial_value=initHyper, learning_rate=learnHyper)
        for i in range(1, n_timesteps+1):
            a = pi.select_action()  # select action
            r = env.act(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(a, r)  # update policy

    return vectorResult


def run_ucb(n_actions, n_timesteps, n_repetitions, cHyper):
    vectorResult = np.zeros(n_timesteps)
    for j in range(n_repetitions):
        env = BanditEnvironment(n_actions=n_actions)
        pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
        for i in range(1, n_timesteps+1):
            a = pi.select_action(c=cHyper, t=i)  # select action
            r = env.act(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(a, r)  # update policy

    return vectorResult


def run_compare(n_actions, n_timesteps, n_repetitions, smoothing_window):
    plotHelper = LearningCurvePlot(
        title="Average reward algorithms over {} repetitions".format(n_repetitions))

    # Assignment 1: e-greedy
    print("Generating e-greedy plot")

    vectorResult = run_egreedy(n_actions, n_timesteps, n_repetitions, 0.1)

    plotHelper.add_curve(
        smooth(vectorResult/float(n_repetitions), window=smoothing_window), label='e-greedy Smooth')

    # Assignment 2: Optimistic init
    print("Generating IO plot")

    vectorResult = run_IO(n_actions, n_timesteps, n_repetitions, 0.1, 5.0)

    plotHelper.add_curve(
        smooth(vectorResult/float(n_repetitions), window=smoothing_window), label='OI Smooth')

    # Assignment 3: UCB
    print("Generating UCB plot")

    vectorResult = run_ucb(n_actions, n_timesteps, n_repetitions, 2.0)

    # plotHelper.add_curve(vectorResult/float(n_repetitions), "UCB")
    plotHelper.add_curve(
        smooth(vectorResult/float(n_repetitions), window=smoothing_window), label='UCB Smooth')

    plotHelper.save("Compare.png")


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
