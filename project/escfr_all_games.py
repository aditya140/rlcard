''' An example of solve Leduc Hold'em with CFR
'''
import torch
import os
import sys

sys.path.append(".")
import numpy as np

import rlcard
from rlcard.agents import ExternalSampling_CFR, RandomAgent
from rlcard import models
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger




def train_leduc():
    # Make environment and enable human mode
    env = rlcard.make('leduc-holdem', config={'seed': 0, 'allow_step_back':True})
    eval_env = rlcard.make('leduc-holdem', config={'seed': 0})

    # Set the iterations numbers and how frequently we evaluate the performance and save model
    evaluate_every = 100
    save_plot_every = 1000
    evaluate_num = 10000
    episode_num = 10000

    # The paths for saving the logs and learning curves
    log_dir = './experiments/leduc_holdem_escfr_result/'

    # Set a global seed
    set_global_seed(0)

    # Initilize CFR Agent
    agent = ExternalSampling_CFR(env)
    agent.load()  # If we have saved model, we first load the model

    # Evaluate CFR against pre-trained NFSP
    eval_env.set_agents([agent, models.load('leduc-holdem-nfsp').agents[0]])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):
        agent.train()
        print('\rIteration {}'.format(episode), end='')
        # Evaluate the performance. Play with NFSP agents.
        if episode % evaluate_every == 0:
            agent.save() # Save model
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('ESCFR')

def train_uno():
    # Make environment and enable human mode
    env = rlcard.make('uno', config={'seed': 0, 'allow_step_back':True})
    eval_env = rlcard.make('uno', config={'seed': 0})

    # Set the iterations numbers and how frequently we evaluate the performance and save model
    evaluate_every = 100
    save_plot_every = 1000
    evaluate_num = 10000
    episode_num = 10000

    # The paths for saving the logs and learning curves
    log_dir = './experiments/uno_escfr_result/'

    # Set a global seed
    set_global_seed(0)

    # Initilize CFR Agent
    agent = ExternalSampling_CFR(env)
    agent.load()  # If we have saved model, we first load the model

    # Evaluate CFR against pre-trained NFSP
    eval_env.set_agents([agent, models.load('uno-nfsp').agents[0]])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):
        agent.train()
        print('\rIteration {}'.format(episode), end='')
        # Evaluate the performance. Play with NFSP agents.
        if episode % evaluate_every == 0:
            agent.save() # Save model
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('ESCFR')

def train_mahjong():
    # Make environment and enable human mode
    env = rlcard.make('mahjong', config={'seed': 0, 'allow_step_back':True})
    eval_env = rlcard.make('mahjong', config={'seed': 0})

    # Set the iterations numbers and how frequently we evaluate the performance and save model
    evaluate_every = 100
    save_plot_every = 1000
    evaluate_num = 10000
    episode_num = 10000

    # The paths for saving the logs and learning curves
    log_dir = './experiments/mahjong_escfr_result/'

    # Set a global seed
    set_global_seed(0)

    # Initilize CFR Agent
    agent = ExternalSampling_CFR(env)
    agent.load()  # If we have saved model, we first load the model

    # Evaluate CFR against pre-trained NFSP
    eval_env.set_agents([agent, models.load('mahjong-nfsp').agents[0]])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):
        agent.train()
        print('\rIteration {}'.format(episode), end='')
        # Evaluate the performance. Play with NFSP agents.
        if episode % evaluate_every == 0:
            agent.save() # Save model
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('ESCFR')



if __name__=="__main__":
    train_leduc()
    # train_uno()
    # train_mahjong()
