import time
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
from collections import deque
from policy import Policy
from tqdm import tqdm


class PolicyLearning:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.policy = self.build_network(state_dim, hidden_dim, action_dim)
        self.optimizer = opt.Adam(self.policy.parameters(), lr=1e-3)
        self.history_ = deque(maxlen=5000)
        self.avg_loss = 0
        self.avg_reward = 0
        self.last_trained = None

    def build_network(self, state_dim, hidden_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def get_action(self, state, player_num):
        self.policy.eval()
        with torch.no_grad():
            s = torch.Tensor(state.to_numpy(player=player_num))
            action = self.policy(s).numpy()
        return action

    def train(self, states, rewards):
        self.policy.train()
        states = torch.stack(states)
        rewards = torch.Tensor(rewards)

        actions = self.policy(states)

        loss = -torch.sum(torch.log(actions) * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.avg_loss = loss.item() * 0.05 + self.avg_loss * 0.95

    def append_history(self, state, reward):
        self.history_.append((torch.Tensor(state), reward))
        self.avg_reward = reward * 0.05 + self.avg_reward * 0.95


class PolicyGradientPolicy(Policy):
    def __init__(
        self, action_space_dim, state_dim, player_num, verbose=False, model=None
    ):
        super().__init__()
        self.verbose = verbose
        self.player_num = player_num
        self.action_space_dim = action_space_dim
        self.model = (
            PolicyLearning(state_dim, hidden_dim=16, action_dim=2)
            if model is None
            else model
        )
        self.last_state = None
        self.last_reward = 0
        if verbose:
            self.pbar = tqdm()
        self.start_time = time.time()

    def get_action(self, state, reward_prev):
        action = self.model.get_action(state, self.player_num)

        if self.last_state is not None:
            self.model.append_history(
                self.last_state, reward_prev - self.model.avg_reward
            )

        self.last_state = state.to_numpy(self.player_num).copy()
        self.last_reward = reward_prev

        if time.time() - self.start_time > 1:
            self.model.train(self.last_state, self.last_reward)
            self.last_trained = time.time()

        return action
