import torch
import os
import sys

sys.path.append(".")
import rlcard
from rlcard.agents import RandomAgent, DQN_agent, DQN_conf
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger


def train_blackjack():
    # Make environment
    env = rlcard.make("blackjack", config={"seed": 0})
    eval_env = rlcard.make("blackjack", config={"seed": 0})

    # Set the iterations numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 1000
    episode_num = 100000

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 100

    # The paths for saving the logs and learning curves
    log_dir = "./experiments/blackjack_results_dqn/"

    # Set a global seed
    set_global_seed(0)

    params = {
        "scope": "DQN-Agent",
        "num_actions": env.action_num,
        "replay_memory_size": memory_init_size,
        "num_states": env.state_shape,
        "discount_factor": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 20000,
        "batch_size": 32,
        "train_every": 1,
        "mlp_layers": [128, 128],
        "lr": 0.0005,
    }

    agent_conf = DQN_conf(**params)
    agent = DQN_agent(agent_conf)

    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])

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
    logger.plot("DQN BLACKJACK")

    # Save model
    save_dir = "models/blackjack_dqn_pytorch"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = agent.get_state_dict()
    print(state_dict.keys())
    torch.save(state_dict, os.path.join(save_dir, "model.pth"))


def train_uno():
    # Make environment
    env = rlcard.make("uno", config={"seed": 0})
    eval_env = rlcard.make("uno", config={"seed": 0})

    # Set the iterations numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 1000
    episode_num = 100000

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 100

    # The paths for saving the logs and learning curves
    log_dir = "./experiments/uno_results_dqn/"

    # Set a global seed
    set_global_seed(0)

    params = {
        "scope": "DQN-Agent",
        "num_actions": env.action_num,
        "replay_memory_size": memory_init_size,
        "num_states": env.state_shape,
        "discount_factor": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 20000,
        "batch_size": 32,
        "train_every": 1,
        "mlp_layers": [512, 512],
        "lr": 0.0005,
    }

    agent_conf = DQN_conf(**params)
    agent = DQN_agent(agent_conf)

    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])

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
    logger.plot("DQN UNO")

    # Save model
    save_dir = "models/uno_dqn_pytorch"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = agent.get_state_dict()
    print(state_dict.keys())
    torch.save(state_dict, os.path.join(save_dir, "model.pth"))


if __name__ == "__main__":
    train_uno()