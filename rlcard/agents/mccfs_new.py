import numpy as np
import collections

import os
import pickle

from limitholdem.utils.utils import *

class MCCFRAgent():
    ''' Implement CFR algorithm
    '''

    def __init__(self, env, model_path='./cfr_model', epsilon=0.1, beta=0.1, tau=0.1):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = False
        self.env = env
        self.model_path = model_path

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0
        self.epsilon = epsilon
        self.beta = beta
        self.tau = tau

    def train(self):
        ''' Do one iteration of CFR
        '''
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.player_num):
            self.env.reset()
            probs = np.ones(self.env.player_num)
            self.traverse_tree(probs, player_id)

        # Update policy
        # self.update_policy()

    def traverse_tree(self, probs, player_id):
        """Traverse the game tree, update the regrets

        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value

        Returns:
            state_utilities (list): The expected utilities for all the players
        """
        if self.env.is_over():
            return self.env.get_payoffs()[player_id]/probs[player_id]
        current_player = self.env.get_player_id()

        # action_utilities = {}
        # # v_σ = 0
        # state_utility = np.zeros(self.env.player_num)
        obs, legal_actions = self.get_state(current_player)

        # if current_player == player_id:
        #     sample_action = choice(legal_actions)
        #     self.env.step(sample_action)
        #     return self.traverse_tree(probs, player_id)

        if obs not in self.regrets:
            # regret to be initialized as 0 for each action
            self.regrets[obs] = np.zeros(self.env.action_num)
        if obs not in self.average_policy:
            # Average policy initialized as 0 for each legal state and action pair
            self.average_policy[obs] = np.zeros(self.env.action_num)

        # σ(I,.) <- regretmatching(r(I,.))
        action_probs = self.regret_matching(obs)
        # if P(h) ∉ i
        if not current_player == player_id:
            for action in legal_actions:
                # s[I,a] <- s[I,a] + (σ(I,a)/q)
                self.average_policy[obs][action] += action_probs[action]/probs[current_player]
            sample_action = np.random.choice(legal_actions)
            self.env.step(sample_action)
            utility = self.traverse_tree(probs, player_id)
            self.env.step_back()
            return utility

        sum_si_b = np.sum(self.average_policy[obs])
        action_utilities = np.zeros(len(action_probs))
        for action in legal_actions:
            ro = max(self.epsilon, self.beta + (self.tau * self.average_policy[obs][action])/(self.beta + sum_si_b))
            if np.random.random() < ro:
                self.env.step(action)
                action_utilities[action] = self.traverse_tree(probs*min(1, ro), player_id)
                self.env.step_back()
        # for action in legal_actions:
        #     action_utilities[action] = action_utilities[action]*action_probs[action]
        for action in legal_actions:
            self.regrets[obs][action] += action_utilities[action] - np.dot(action_utilities, action_probs)

        self.policy[obs] = action_utilities
        return np.dot(action_utilities, action_probs)

    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def regret_matching(self, obs):
        """Apply regret matching

        Args:
            obs (string): The state_str
        """

        regret = self.regrets[obs]
        # positive_regret_sum =  Σ_a (R)^T,+(I,a)
        positive_regret_sum = sum([r for r in regret if r > 0])

        # action probs = σ^T+1 (I,.)
        action_probs = np.zeros(self.env.action_num)
        if positive_regret_sum > 0:
            for action in range(self.env.action_num):
                # σ^T+1 (I,a) = R^T,+(I,a) /  Σ_a (R)^T,+(I,a)
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.action_num):
                # σ^T+1 (I,a) = 1/|A(I)|
                action_probs[action] = 1.0 / self.env.action_num

        return action_probs

    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        if obs not in policy.keys():
            action_probs = np.array([1.0/self.env.action_num for _ in range(self.env.action_num)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
        '''
        probs = self.action_probs(state['obs'].tostring(), state['legal_actions'], self.average_policy)
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), state['legal_actions']

    def save(self):
        ''' Save model
        '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()

