import numpy as np


class Actions:
    def __init__(self, num_players: int, action_dim: int):
        self.num_players = num_players
        self.action_dim = action_dim
        self.actions = np.zeros((num_players, action_dim))

    def set_action(self, player_index: int, action: np.ndarray):
        if action.shape[0] != self.action_dim:
            raise ValueError("Invalid action dimensions")
        self.actions[player_index] = action

    def get_action(self, player_index: int) -> np.ndarray:
        return self.actions[player_index]

    def to_numpy(self) -> np.ndarray:
        return self.actions.flatten()
