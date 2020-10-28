import torch
import numpy as np
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rlcard.utils.utils import remove_illegal, Memory


class DQN_conf:
    scope = "DQN"
    num_states = None
    num_actions = None
    replay_memory_size = None  # Size of the replay memory
    replay_memory_init_size = 100
    update_target_estimator_every = 1000
    discount_factor = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_steps = 20000
    batch_size = 32
    train_every = 1
    mlp_layers = None
    lr = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DQN_agent:
    def __init__(self, conf):
        self.use_raw = False
        self.conf = conf

        self.scope = self.conf.scope

        self.total_t = 1  # total timesteps
        self.train_t = 0  # train timesteps

        # Epsilon (Exploreability of model)
        self.epsilon_decay_steps = self.conf.epsilon_decay_steps
        self.epsilons = np.linspace(
            self.conf.epsilon_start,
            self.conf.epsilon_end,
            self.conf.epsilon_decay_steps,
        )

        # Q-Estimators
        self.q_estimator = Estimator(
            num_states=self.conf.num_states,
            num_actions=self.conf.num_actions,
            learning_rate=self.conf.lr,
            layers=self.conf.mlp_layers,
            device=self.conf.device,
        )
        # self.target_estimator = Estimator(num_states = self.conf.num_states,num_actions = self.conf.num_actions, learning_rate = self.conf.lr, layers = self.conf.mlp_layers)

        # Memory
        self.memory = Memory(
            memory_size=self.conf.replay_memory_size, batch_size=self.conf.batch_size
        )

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        # Save to memory
        self.feed_memory(state["obs"], action, reward, next_state["obs"], done)
        self.total_t += 1

        tmp = self.total_t - self.conf.replay_memory_init_size
        if tmp >= 0 and tmp % self.conf.train_every == 0:
            self.train()

    def train(self):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.memory.sample()

        q_values_next = self.q_estimator.predict_nograd(next_state_batch)
        best_actions = np.argmax(q_values_next, axis=1)

        target_batch = (
            reward_batch
            + np.invert(done_batch).astype(np.float32)
            * self.conf.discount_factor
            * q_values_next[np.arange(self.conf.batch_size), best_actions]
        )

        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        print(
            "\rINFO - Agent {}, step {}, rl-loss: {}".format(
                self.scope, self.total_t, loss
            ),
            end="",
        )

        self.train_t += 1

    def step(self, states):
        A = self.predict(states["obs"])
        A = remove_illegal(A, states["legal_actions"])
        action = np.random.choice(np.arange(len(A)), p=A)
        return action

    def eval_step(self, states):
        q_values = self.q_estimator.predict_nograd(np.expand_dims(states["obs"], 0))[0]
        probs = remove_illegal(np.exp(q_values), states["legal_actions"])
        best_action = np.argmax(probs)
        return best_action, probs

    def predict(self, state):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        A = (
            np.ones(self.conf.num_actions, dtype=float)
            * epsilon
            / self.conf.num_actions
        )
        q_values = self.q_estimator.predict_nograd(np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += 1.0 - epsilon
        return A

    def feed_memory(self, state, action, reward, next_state, done):
        self.memory.save(state, action, reward, next_state, done)

    def get_state_dict(self):
        q_value = self.q_estimator.qnet.state_dict()
        return q_value

    def load(self, model_weights):
        self.q_estimator.qnet.load_state_dict(model_weights)


class Estimator(object):
    def __init__(self, num_states, num_actions, learning_rate, layers, device):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.layers = layers
        self.device = device

        self.qnet = EstimatorNet(num_states, num_actions, layers)
        self.qnet = self.qnet.to(device)
        self.qnet.eval()

        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = optim.AdamW(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_action = self.qnet(s).cpu().numpy()
        return q_action

    def update(self, s, a, y):

        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, action_num)
        q_action = self.qnet(s)

        # (batch, action_num) -> (batch, )
        Q = torch.gather(q_action, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # Backprop
        batch_loss = self.loss_fn(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss


class EstimatorNet(nn.Module):
    def __init__(self, num_states, num_actions, layers):
        super(EstimatorNet, self).__init__()
        self.num_actions = num_actions
        self.num_states = num_states
        self.layers = layers

        layer_dims = [np.prod(self.num_states)] + self.layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        return self.fc_layers(s)
