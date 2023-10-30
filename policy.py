import numpy as np
import pygame


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

    def get_action(self, state) -> np.ndarray:
        keys = pygame.key.get_pressed()
        action = np.zeros(self.action_space_dim)

        if keys[pygame.K_UP]:
            action[1] = -1
        if keys[pygame.K_DOWN]:
            action[1] = 1
        if keys[pygame.K_LEFT]:
            action[0] = -1
        if keys[pygame.K_RIGHT]:
            action[0] = 1

        return action
