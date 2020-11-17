import torch
import os
import sys

sys.path.append(".")
import rlcard
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger


import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from rlcard.agents import RandomAgent



def train_blackjack():
        
    # Make environment
    env = rlcard.make('blackjack', config={'seed': 0})
    eval_env = rlcard.make('blackjack', config={'seed': 0})

    # Set the iterations numbers and how frequently we evaluate performance
    evaluate_every = 100
    evaluate_num = 10000
    episode_num = 3000

    # The intial memory size
    memory_init_size = 100

    # Train the agent every X steps
    train_every = 1

    # The paths for saving the logs and learning curves
    log_dir = './experiments/blackjack_dqn_result/'

    # Set a global seed
    set_global_seed(0)

    with tf.Session() as sess:

        # Initialize a global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set up the agents
        agent = DQNAgent(sess,
                        scope='dqn',
                        action_num=env.action_num,
                        replay_memory_init_size=memory_init_size,
                        train_every=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[10,10])
        env.set_agents([agent])
        eval_env.set_agents([agent])

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Initialize a Logger to plot the learning curve
        logger = Logger(log_dir)

        for episode in range(episode_num):

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

        # Close files in the logger
        logger.close_files()

        # Plot the learning curve
        logger.plot('DQN')
        
        # Save model
        save_dir = 'models/blackjack_dqn'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(save_dir, 'model'))


def train_uno():
    
    # Make environment
    env = rlcard.make('uno', config={'seed': 0})
    eval_env = rlcard.make('uno', config={'seed': 0})

    # Set the iterations numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 1000
    episode_num = 3000

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 1

    # The paths for saving the logs and learning curves
    log_dir = './experiments/uno_dqn_result/'

    # Set a global seed
    set_global_seed(0)

    with tf.Session() as sess:

        # Initialize a global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set up the agents
        agent = DQNAgent(sess,
                        scope='dqn',
                        action_num=env.action_num,
                        replay_memory_size=20000,
                        replay_memory_init_size=memory_init_size,
                        train_every=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[512,512])
        random_agent = RandomAgent(action_num=eval_env.action_num)
        env.set_agents([agent, random_agent])
        eval_env.set_agents([agent, random_agent])

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Init a Logger to plot the learning curve
        logger = Logger(log_dir)

        for episode in range(episode_num):

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

        # Close files in the logger
        logger.close_files()

        # Plot the learning curve
        logger.plot('DQN')
        
        # Save model
        save_dir = 'models/uno_dqn'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(save_dir, 'model'))
        

def train_leduc_holdem_poker():
        
    # Make environment
    env = rlcard.make('leduc-holdem', config={'seed': 0})
    eval_env = rlcard.make('leduc-holdem', config={'seed': 0})

    # Set the iterations numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 10000
    episode_num = 3000

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 1

    # The paths for saving the logs and learning curves
    log_dir = './experiments/leduc_holdem_dqn_result/'

    # Set a global seed
    set_global_seed(0)

    with tf.Session() as sess:

        # Initialize a global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set up the agents
        agent = DQNAgent(sess,
                        scope='dqn',
                        action_num=env.action_num,
                        replay_memory_init_size=memory_init_size,
                        train_every=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[128,128])
        random_agent = RandomAgent(action_num=eval_env.action_num)
        env.set_agents([agent, random_agent])
        eval_env.set_agents([agent, random_agent])

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Init a Logger to plot the learning curve
        logger = Logger(log_dir)

        for episode in range(episode_num):

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

        # Close files in the logger
        logger.close_files()

        # Plot the learning curve
        logger.plot('DQN')

        # Save model
        save_dir = 'models/leduc_holdem_dqn'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(save_dir, 'model'))


def train_mahjong():
        
    # Make environment
    env = rlcard.make('mahjong', config={'seed': 0})
    eval_env = rlcard.make('mahjong', config={'seed': 0})

    # Set the iterations numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 1000
    episode_num = 3000

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 1

    # The paths for saving the logs and learning curves
    log_dir = './experiments/mahjong_dqn_result/'

    # Set a global seed
    set_global_seed(0)

    with tf.Session() as sess:

        # Initialize a global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set up the agents
        agent = DQNAgent(sess,
                        scope='dqn',
                        action_num=env.action_num,
                        replay_memory_size=20000,
                        replay_memory_init_size=memory_init_size,
                        train_every=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[512,512])
        random_agent = RandomAgent(action_num=eval_env.action_num)
        env.set_agents([agent, random_agent, random_agent, random_agent])
        eval_env.set_agents([agent, random_agent, random_agent, random_agent])

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Init a Logger to plot the learning curve
        logger = Logger(log_dir)

        for episode in range(episode_num):

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

        # Close files in the logger
        logger.close_files()

        # Plot the learning curve
        logger.plot('DQN')
        
        # Save model
        save_dir = 'models/mahjong_dqn'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(save_dir, 'model'))
        


if __name__ == "__main__":
    # train_uno()
    # train_blackjack()
    # train_leduc_holdem_poker()
    train_mahjong()
