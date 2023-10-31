import numpy as np
import pygame
from state import State
import torch
import torch.optim as opt
import torch.nn as nn


class Policy:
    def get_action(self, state):
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, action_space_dim: int):
        self.action_space_dim = action_space_dim

    def get_action(self, state) -> np.ndarray:
        return np.zeros(self.action_space_dim)


class KeyboardControlPolicy(Policy):
    def __init__(self, action_space_dim: int):
        self.action_space_dim = action_space_dim

    def get_action(self, state, reward) -> np.ndarray:
        keys = pygame.key.get_pressed()
        action = np.zeros(self.action_space_dim)

        x_move = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        y_move = float(keys[pygame.K_DOWN]) - float(keys[pygame.K_UP])

        action[0] = x_move
        action[1] = y_move
        # action = action / np.linalg.norm(action)
        return action


class FollowPolicy(Policy):
    def __init__(self, action_space_dim: int, player_num=1):
        super().__init__()
        self.player_num = player_num
        self.action_space_dim = action_space_dim

    def get_action(self, state: State, reward) -> np.ndarray:
        action = np.zeros(self.action_space_dim)
        ball_direction = state.ball_position - state.player_positions[self.player_num]
        action[0] = ball_direction[0] / np.linalg.norm(ball_direction)
        action[1] = ball_direction[1] / np.linalg.norm(ball_direction)
        action = (
            action
            / np.abs(
                state.ball_position - state.player_positions[self.player_num]
            ).mean()
        )
        action = action / (np.linalg.norm(action) + 1e-7) * 1
        return action


class LinearPolicy(Policy):
    def __init__(self, action_space_dim: int, state_dim: int, player_num=1):
        super().__init__()
        self.player_num = player_num
        self.action_space_dim = action_space_dim
        self.lin = nn.Linear(state_dim, action_space_dim, bias=True)
        optimizer = opt.Adam(self.lin.parameters(), lr=1e-4)
        self.history_ = []
        self.value = 0

    def get_action(self, state: State, reward_prev: float) -> np.ndarray:
        with torch.no_grad():
            s = torch.Tensor(state.to_numpy())
            a = torch.tanh(self.lin(s))
        return a.numpy()
