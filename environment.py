import time
import numpy as np
import pickle
from state import State
from policy import Policy
from action import Actions


class Environment:
    def __init__(
        self,
        num_players: int,
        policies: list[Policy],
        tick_rate=60,
        target_framerate=60,
    ):
        self.num_players = num_players
        self.policies = policies
        self.state_history = []
        self.action_history = []
        self.state = State(num_players)
        self.tick_rate = tick_rate
        self.time_per_tick = 1.0 / tick_rate
        self.target_framerate = target_framerate
        self.delta_time = 1.0 / target_framerate
        self.prev_positions = self.state.player_positions.copy()

    def save_history(
        self, state_path="state_history.pkl", action_path="action_history.pkl"
    ):
        with open(state_path, "wb") as f:
            pickle.dump(self.state_history, f)
        with open(action_path, "wb") as f:
            pickle.dump(self.action_history, f)

    def load_history(
        self, state_path="state_history.pkl", action_path="action_history.pkl"
    ):
        with open(state_path, "rb") as f:
            self.state_history = pickle.load(f)
        with open(action_path, "rb") as f:
            self.action_history = pickle.load(f)

    def run_game(self, run_time=30):
        end_time = time.time() + run_time
        last_time = time.time()
        next_tick = time.time()

        while time.time() < end_time:
            current_time = time.time()
            self.delta_time = current_time - last_time

            self.tick(self.delta_time)

            next_tick += self.time_per_tick
            sleep_time = next_tick - current_time

            if sleep_time > 0:
                time.sleep(sleep_time)

            last_time = current_time

    def get_actions(self, delta_time):
        actions = Actions(self.num_players, action_dim=2)
        ball_speed = np.linalg.norm(self.state.ball_velocity)
        for player_id in range(2):
            ball_distance = np.linalg.norm(
                self.state.ball_position - self.state.player_positions[player_id]
            )
            ball_proximity_reward = (
                1 if ball_distance < 0.2 else (0.5 if ball_distance < 0.4 else 0)
            )
            position_change_reward = np.abs(
                self.state.player_positions - self.prev_positions
            ).mean()
            reward = np.log(ball_speed + 1)
            # reward += np.log(ball_distance + 1) * 0.1
            actions.set_action(
                player_id, self.policies[player_id].get_action(self.state, reward)
            )
        return actions

    def tick(self, delta_time):
        actions = self.get_actions(delta_time)
        self.prev_positions = self.state.player_positions.copy()
        self.state.update(actions, delta_time)
        self.state_history.append(self.state.to_numpy())
        self.action_history.append(actions.to_numpy())

    def render(self, renderer):
        renderer.render(self.state.to_numpy())


class RenderVideo:
    def __init__(self, game_states, resolution, framerate, video_path, verbose=False):
        pass

    def render(self):
        pass
