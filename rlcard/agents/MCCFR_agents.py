import numpy as np
import collections

import os
import pickle

from rlcard.utils.utils import *


class ChanceSampling_CFR:
    """Implementation of CFR algorithm"""

    def __init__(self, env, model_path="./ChanceSampling_cfr_model"):
        """Initialize Model

        Args:
            env : Env Clas
            model_path (str, optional): [description]. Defaults to './vanila_cfr_model'.
        """

        self.use_raw = False
        self.env = env
        self.model_path = model_path

        # Policy = σ
        # A policy is a dict state_str -> action probability
        self.policy = collections.defaultdict(list)

        # average_policy = s
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0

    def train(self):
        """ One iteration of CRF"""
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.player_num):
            self.env.reset()
            # probs = π
            probs = np.ones(self.env.player_num)
            self.traverse_tree(probs, player_id)
        # Update policy
        self.update_policy()

    def traverse_tree(self, probs, player_id):
        """
        if  terminal node return utility = payoff
        """
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        # action_utilities = v_σ(I->a)
        action_utilities = {}

        # state_utility = v_σ
        state_utility = np.zeros(self.env.player_num)

        # obs = h, legal_actions = A(I)
        obs, legal_actions = self.get_state(current_player)

        # action probs = σ(I,.)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        """
        σ^t(I) <- RegretMatching(r_I)
        v_σ <- 0
        v_σ_(I->a)[a] <- 0 for all a ∈ A(I)
        """
        for action in legal_actions:
            # action prob = σ(I,a)
            action_prob = action_probs[action]
            new_probs = probs.copy()
            # σ(I,a) * π
            new_probs[current_player] *= action_prob

            # Keep traversing the child state
            # Each chance node is sampled internally by the environment
            self.env.step(action)
            # Utility = v_σ(I->a)[a]
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            # State utility = v_σ
            # v_σ = v_σ + σ(I,a)*v_σ(I->a)[a]
            state_utility += action_prob * utility

            action_utilities[action] = utility

        """
        if P(h) != i:
            return v_σ 
        """
        if not current_player == player_id:
            return state_utility

        """
        if P(h) == i:
            for a ∈ A(I):
                r_I[a] <- r_I[a] + π_(-i) (v_σ(I->a)[a] - v_σ)
                s[I,a] <- s[I,a] + iter * π_(i) * σ(I,a)
        """

        # If it is current player, we record the policy and compute regret
        # player_prob = π_(i)
        player_prob = probs[current_player]
        # counterfactual_prob = π_(-i)
        counterfactual_prob = np.prod(probs[:current_player]) * np.prod(
            probs[current_player + 1 :]
        )
        # player_state_utility = v_σ
        player_state_utility = state_utility[current_player]
        if obs not in self.regrets:
            # regret to be initialized as 0 for each action
            self.regrets[obs] = np.zeros(self.env.action_num)
        if obs not in self.average_policy:
            # Average policy initialized as 0 for each legal state and action pair
            self.average_policy[obs] = np.zeros(self.env.action_num)
        for action in legal_actions:
            # action prob = σ(I,a)
            action_prob = action_probs[action]

            # regret = π_(-i) * (v_σ(I->a)[a] - v_σ)
            regret = counterfactual_prob * (
                action_utilities[action][current_player] - player_state_utility
            )
            # r[I,a] = r[I,a] + π_(-i) * (v_σ(I->a)[a] - v_σ)
            self.regrets[obs][action] += regret

            # s[I,a] = s[I,a] + iter * π_(i) * σ(I,a)
            self.average_policy[obs][action] += player_prob * action_prob
        # return v_σ
        return state_utility

    def regret_matching(self, obs):
        """Regret Matching using
                        { R^T,+(I,a) /  Σ_a (R)^T,+(I,a)    if Σ_a (R)^T,+(I,a)>0
        σ^(T+1) (I,a) = {
                        { 1/|A(I)|          else

        Args:
            obs (string):
        Returns:
            action_probability
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

    def update_policy(self):
        """Update policy based on the current regrets"""
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def action_probs(self, obs, legal_actions, policy):
        """Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        """
        if obs not in policy.keys():
            action_probs = np.array(
                [1.0 / self.env.action_num for _ in range(self.env.action_num)]
            )
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def eval_step(self, state):
        """Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
        """
        probs = self.action_probs(
            state["obs"].tostring(), state["legal_actions"], self.average_policy
        )
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def get_state(self, player_id):
        """Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        """
        state = self.env.get_state(player_id)
        return state["obs"].tostring(), state["legal_actions"]

    def save(self):
        """Save model"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, "policy.pkl"), "wb")
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(
            os.path.join(self.model_path, "average_policy.pkl"), "wb"
        )
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, "regrets.pkl"), "wb")
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, "iteration.pkl"), "wb")
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        """Load model"""
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, "policy.pkl"), "rb")
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(
            os.path.join(self.model_path, "average_policy.pkl"), "rb"
        )
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, "regrets.pkl"), "rb")
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, "iteration.pkl"), "rb")
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()


class ExternalSampling_CFR:
    """Implementation of External Sampling CFR algorithm
    External Sampling with Stochastically-Weighted Averaging



    Initialize: ∀I ∈ I,∀a ∈ A(I) : rI[a] ← sI[a] ← 0
    ExternalSampling(h,i):
        if h ∈ Z then return ui(h)
        if P (h) = c then sample a′ and return ExternalSampling(ha′ , i)
        Let I be the information set containing h
        σ(I) ← RegretMatching(rI )
        if P(I)=i then
            Let u be an array indexed by actions and uσ ← 0
            for a ∈ A(I) do
                u[a] ← ExternalSampling(ha, i)
                uσ ←uσ +σ(I,a)·u[a]
            for a ∈ A(I) do
                By Equation 4.20, compute r ̃(I, a) ← u[a] − uσ
                r I [ a ] ← r I [ a ] + r ̃ ( I , a )
            return uσ
        else
            Sample action a′ from σ(I)
            u ← ExternalSampling(ha′, i)
            for a ∈ A(I) do
                sI [a] ← sI [a] + σ(I, a)
            return u



    """

    def __init__(self, env, model_path="./external_sampling_cfr_model"):
        """Initialize Model

        Args:
            env : Env Clas
            model_path (str, optional): [description]. Defaults to './vanila_cfr_model'.
        """

        self.use_raw = False
        self.env = env
        self.model_path = model_path

        # Policy = σ
        # A policy is a dict state_str -> action probability
        self.policy = collections.defaultdict(list)

        # average_policy = s
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0

    def train(self):
        """ One iteration of CRF"""
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.player_num):
            self.env.reset()
            self.traverse_tree(player_id)
        # Update policy
        self.update_policy()

    def traverse_tree(self, player_id):
        """
        if  terminal node return utility = payoff
        """
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        # state_utility = v_σ
        state_utility = np.zeros(self.env.player_num)

        # obs = h, legal_actions = A(I)
        obs, legal_actions = self.get_state(current_player)

        if obs not in self.regrets:
            # regret to be initialized as 0 for each action
            self.regrets[obs] = np.zeros(self.env.action_num)
        if obs not in self.average_policy:
            # Average policy initialized as 0 for each legal state and action pair
            self.average_policy[obs] = np.zeros(self.env.action_num)

        # σ(I,.) = regret_matching(r_I)
        self.policy[obs] = self.regret_matching(obs)

        # action probs = σ(I,.)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        # if P(h) == i
        if current_player == player_id:

            # Let u be an array indexed by actions and  u_σ <- 0

            action_utility = {}
            state_utility = np.zeros(self.env.player_num)

            for action in legal_actions:
                action_prob = action_probs[action]
                self.env.step(action)

                # u[a] <- ExternalSampling(ha,i)
                utility = self.traverse_tree(player_id)
                action_utility[action] = utility
                self.env.step_back()

                # u_σ += σ(I,a) * u[a]
                state_utility += action_prob * utility

            for action in legal_actions:

                # player_state_utility = v_σ
                player_state_utility = state_utility[current_player]

                # r(I,a) <- u[a] - u_σ
                regret = (
                    action_utility[action][current_player]
                    - state_utility[current_player]
                )
                # r_I[a] += r(I,a)
                self.regrets[obs][action] += regret

            return state_utility

        # if P(h) != i:
        else:
            # sample one single action
            sampled_action = random.choice(legal_actions)
            self.env.step(sampled_action)
            utility = self.traverse_tree(player_id)
            self.env.step_back()
            for action in legal_actions:
                # s_I[a] += σ(I,a)
                self.average_policy[obs][action] += action_probs[action]
            return utility

    def regret_matching(self, obs):
        """Regret Matching using
                        { R^T,+(I,a) /  Σ_a (R)^T,+(I,a)    if Σ_a (R)^T,+(I,a)>0
        σ^(T+1) (I,a) = {
                        { 1/|A(I)|          else

        Args:
            obs (string):
        Returns:
            action_probability
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

    def update_policy(self):
        """Update policy based on the current regrets"""
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def action_probs(self, obs, legal_actions, policy):
        """Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        """
        if obs not in policy.keys():
            action_probs = np.array(
                [1.0 / self.env.action_num for _ in range(self.env.action_num)]
            )
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def eval_step(self, state):
        """Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
        """
        probs = self.action_probs(
            state["obs"].tostring(), state["legal_actions"], self.average_policy
        )
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def get_state(self, player_id):
        """Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        """
        state = self.env.get_state(player_id)
        return state["obs"].tostring(), state["legal_actions"]

    def save(self):
        """Save model"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, "policy.pkl"), "wb")
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(
            os.path.join(self.model_path, "average_policy.pkl"), "wb"
        )
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, "regrets.pkl"), "wb")
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, "iteration.pkl"), "wb")
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        """Load model"""
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, "policy.pkl"), "rb")
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(
            os.path.join(self.model_path, "average_policy.pkl"), "rb"
        )
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, "regrets.pkl"), "rb")
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, "iteration.pkl"), "rb")
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()


class OutcomeSampling_CFR:
    """Implementation of Outcome Sampling CFR algorithm


    Initialize:∀I∈I:cI ←0
    Initialize: ∀I ∈ I,∀a ∈ A(I) : rI[a] ← sI[a] ← 0

    OutcomeSampling(h, i, t, πi, π−i, s):
        if h ∈ Z then return (ui(h)/s, 1)
        if P (h) = c then sample a′ and return OutcomeSampling(ha′ , i, t, πi , π−i , s)
        Let I be the information set containing h
        σ(I) ← RegretMatching(rI )
        Let σ′(I) be a sampling distribution at I
        if P(I)=ithenσ′(I)←ε·Unif(I)+(1−ε)σ(I)
        else σ′(I) ← σ(I)
        Sample an action a′ with probability σ′(I, a)
        if P(I)=i then
            (u, πtail) ← OutcomeSampling(ha′, i, t, πi · σ(I, a), π−i, s · σ′(I, a))
            for a ∈ A(I) do
                W ← u · π−i
                Compute r ̃(I,a) from Equation 4.12 ifa=a′ else Equation4.15
                r I [ a ] ← r I [ a ] + r ̃ ( I , a )
        else
            (u, πtail) ← OutcomeSampling(ha′, i, t, πi, π−i · σ(I, a), s · σ′(I, a))
            for a ∈ A(I) do
                sI[a]←sI[a]+(t−cI)·π−i ·σ(I,a)
                cI ← t
        return (u, πtail · σ(I, a))
    """

    def __init__(self, env, epsilon = 0.5, model_path="./OutcomeSampling_cfr_model"):
        """Initialize Model

        Args:
            env : Env Class
            model_path (str, optional): [description]. Defaults to './OutcomeSampling_cfr_model'.
        """

        self.use_raw = False
        self.env = env
        self.model_path = model_path

        # Policy = σ
        # A policy is a dict state_str -> action probability
        self.policy = collections.defaultdict(list)

        # average_policy = s
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        # counter for each Information set
        self.counter = {}

        self.epsilon = epsilon
        self.iteration = 0

    def train(self):
        """ One iteration of CRF"""
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.player_num):
            self.env.reset()
            # probs = [π_i,π_-i]
            probs = np.ones(self.env.player_num)
            s = 1
            self.traverse_tree(probs, player_id, s)
        # Update policy
        self.update_policy()

    def traverse_tree(self, probs, player_id, s):
        """
        if  terminal node return utility = payoff
        """
        if self.env.is_over():
            return (self.env.get_payoffs() / s, 1)

        current_player = self.env.get_player_id()
        next_player = int(not (current_player))
        # state_utility = v_σ
        state_utility = np.zeros(self.env.player_num)

        # obs = h, legal_actions = A(I)
        obs, legal_actions = self.get_state(current_player)

        if obs not in self.regrets:
            # regret to be initialized as 0 for each action
            self.regrets[obs] = np.zeros(self.env.action_num)
        if obs not in self.average_policy:
            # Average policy initialized as 0 for each legal state and action pair
            self.average_policy[obs] = np.zeros(self.env.action_num)
        if obs not in self.counter.keys():
            self.counter[obs] = 0
        # σ(I,.) = regret_matching(r_I)
        self.policy[obs] = self.regret_matching(obs)

        # action probs = σ(I,.)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        if current_player == player_id:
            sampling_dist = (
                self.epsilon
                * remove_illegal(np.ones(self.env.action_num), legal_actions)
                / self.env.action_num
            ) + (1 - self.epsilon) * (action_probs)
        else:
            sampling_dist = action_probs

        sampled_action = np.random.choice(
            range(self.env.action_num), p=remove_illegal(sampling_dist, legal_actions)
        )

        if current_player == player_id:

            # action prob = σ(I,a)
            action_prob = action_probs[sampled_action]

            new_probs = probs.copy()
            # σ(I,a) * π
            new_probs[current_player] *= action_prob

            self.env.step(sampled_action)
            (u, pi_tail) = self.traverse_tree(
                new_probs, player_id, s * sampling_dist[sampled_action]
            )
            self.env.step_back()

            for action in legal_actions:
                W = u * probs[next_player]
                if action == sampled_action:
                    # W ·(πσ(z[I]a,z)−πσ(z[I],z))
                    regret = W[current_player] * (pi_tail - probs[current_player])
                else:
                    regret = -1 * W[current_player] * probs[current_player]

                self.regrets[obs][action] += regret
        else:

            # action prob = σ(I,a)
            action_prob = action_probs[sampled_action]

            new_probs = probs.copy()
            # σ(I,a) * π
            new_probs[current_player] *= action_prob
            self.env.step(sampled_action)
            (u, pi_tail) = self.traverse_tree(
                new_probs, player_id, s * sampling_dist[sampled_action]
            )
            self.env.step_back()

            for action in legal_actions:
                self.average_policy[obs][action] += (
                    (self.iteration - self.counter[obs]) * probs[next_player] * action_prob
                )
                self.counter[obs] = self.iteration
        return (u, pi_tail * action_prob)

    def regret_matching(self, obs):
        """Regret Matching using
                        { R^T,+(I,a) /  Σ_a (R)^T,+(I,a)    if Σ_a (R)^T,+(I,a)>0
        σ^(T+1) (I,a) = {
                        { 1/|A(I)|          else

        Args:
            obs (string):
        Returns:
            action_probability
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

    def update_policy(self):
        """Update policy based on the current regrets"""
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def action_probs(self, obs, legal_actions, policy):
        """Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        """
        if obs not in policy.keys():
            action_probs = np.array(
                [1.0 / self.env.action_num for _ in range(self.env.action_num)]
            )
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def eval_step(self, state):
        """Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
        """
        probs = self.action_probs(
            state["obs"].tostring(), state["legal_actions"], self.average_policy
        )
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def get_state(self, player_id):
        """Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        """
        state = self.env.get_state(player_id)
        return state["obs"].tostring(), state["legal_actions"]

    def save(self):
        """Save model"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, "policy.pkl"), "wb")
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(
            os.path.join(self.model_path, "average_policy.pkl"), "wb"
        )
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, "regrets.pkl"), "wb")
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, "iteration.pkl"), "wb")
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        """Load model"""
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, "policy.pkl"), "rb")
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(
            os.path.join(self.model_path, "average_policy.pkl"), "rb"
        )
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, "regrets.pkl"), "rb")
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, "iteration.pkl"), "rb")
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()
