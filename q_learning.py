import time
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
from collections import deque
import random
from policy import Policy
from state import State
from tqdm import tqdm

ACTIONS = [
    np.array([-1, -1]),
    np.array([-1, 1]),
    np.array([1, -1]),
    np.array([1, 1]),
    np.array([1, 0]),
    np.array([-1, 0]),
    np.array([0, -1]),
    np.array([0, 1]),
]


class DeepQLearning:
    def __init__(self, state_dim, hidden_dim):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim, bias=True),
            nn.ELU(),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, len(ACTIONS), bias=True),
            # nn.Tanh(),
        )
        self.optimizer = opt.Adam(self.policy.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.history_ = deque(maxlen=5000)
        self.avg_loss = 0
        self.avg_reward = 0
        self.last_trained = None

    def get_q_values(self, state, player_num) -> torch.Tensor:
        with torch.no_grad():
            self.policy.eval()
            s = torch.Tensor(state.to_numpy(player=player_num))
            q_values = self.policy(s)
            return q_values

    def train(self):
        BATCH_SIZE = 512
        if len(self.history_) < BATCH_SIZE:  # Don't update if not enough samples
            return

        # Sample random experiences from the memory
        minibatch = random.sample(self.history_, BATCH_SIZE)

        states, actions, rewards = zip(*minibatch)

        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.Tensor(rewards)

        # Compute the Q-values from the network for current states
        self.policy.train()
        current_q_values = self.policy(states)

        # Compute the Q-values for the next states
        self.policy.train()
        next_q_values = self.policy(states)
        max_next_q_values, _ = torch.max(next_q_values, dim=1)

        # Compute target Q-values: immediate rewards + (discount factor * max Q-value for next state)
        discount_factor = 0.9
        targets = rewards + discount_factor * max_next_q_values
        # targets = rewards

        # Select the Q-values for the taken actions
        picked_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute the loss
        loss = self.criterion(picked_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.avg_loss == 0:
            self.avg_loss = loss.item()
        else:
            self.avg_loss = loss.item() * 0.05 + self.avg_loss * 0.95

    def append_history(self, state, action, reward):
        self.history_.append((torch.Tensor(state), action, reward))
        if self.avg_reward == 0:
            self.avg_reward = reward
        else:
            self.avg_reward = reward * 0.05 + self.avg_reward * 0.95


class QPolicy(Policy):
    def __init__(
        self,
        action_space_dim: int,
        state_dim: int,
        player_num,
        verbose=False,
        model: DeepQLearning | None = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.player_num = player_num
        self.action_space_dim = action_space_dim
        if model is None:
            self.model = DeepQLearning(state_dim, hidden_dim=16)
        else:
            self.model = model
        self.value = 0
        self.last_updated = None
        self.last_action = None
        self.last_state = None
        self.last_reward = 0
        self.started_exploring = False
        self.explore_action = None
        if verbose:
            self.pbar = tqdm()
        self.start_time = time.time()

    def get_action(self, state: State, reward_prev: float) -> np.ndarray:
        q_values = self.model.get_q_values(state, self.player_num)
        if self.started_exploring is None:
            # Exploit
            _, action_idx = torch.max(q_values, dim=0)
            action_idx = int(action_idx.item())
            if np.random.uniform(0, 1) < 0.01:
                self.started_exploring = time.time()
                self.random_a = None
        else:
            # Explore
            if self.explore_action is None:
                random_a = np.random.choice(len(ACTIONS))
                self.explore_duration = random_a
                action_idx = random_a
            else:
                action_idx = self.explore_action
            self.explore_duration = time.time() - self.started_exploring
            if self.explore_duration > 2:
                self.started_exploring = None
                self.explore_duration = 0
        action = ACTIONS[action_idx]

        explore_action = (
            ACTIONS[self.explore_action] if self.explore_action is not None else "-"
        )
        summary = (
            f"reward: {self.model.avg_reward:0.3f} loss: {100*self.model.avg_loss:0.3f} explore: { explore_action} q:"
            + ", ".join(["{:.2f}".format(q) for q in q_values.numpy()])
        )
        if self.verbose:
            self.pbar.set_postfix_str(summary)
            self.pbar.update()
        if self.last_action is not None and self.last_state is not None:
            self.model.append_history(
                self.last_state, self.last_action, reward_prev - self.model.avg_reward
            )
        self.last_action = action_idx
        self.last_state = state.to_numpy(self.player_num).copy()
        self.last_reward = reward_prev
        if time.time() - self.start_time > 1:
            self.model.train()
            self.last_trained = time.time()
        action = action / (np.linalg.norm(action) + 1e-7)
        return action
