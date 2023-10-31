import cv2
import logging
import subprocess
import numpy as np
from state import PLAYER_RADIUS, BALL_RADIUS


class RenderMP4:
    def __init__(self, game_states, resolution, framerate, video_path, verbose=False):
        self.game_states = game_states
        self.resolution = resolution
        self.framerate = framerate
        self.video_path = video_path
        self.verbose = verbose

    def decompose_state(self, state):
        # Directly use attributes from the State object
        player_positions = state.player_positions
        ball_position = state.ball_position
        return player_positions, ball_position

    def draw_frame(self, state, frame):
        player_positions, ball_position = self.decompose_state(state)
        height, width, _ = frame.shape

        for position in player_positions:
            x, y = (position * (width, height)).astype(int)
            cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)

        ball_x, ball_y = (ball_position * (width, height)).astype(int)
        cv2.circle(frame, (ball_x, ball_y), 10, (0, 0, 255), -1)

    def render(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(self.video_path, fourcc, self.framerate, self.resolution)

        empty_frame = np.zeros(
            (self.resolution[1], self.resolution[0], 3), dtype=np.uint8
        )

        for state in self.game_states:
            frame = empty_frame.copy()
            self.draw_frame(state, frame)
            out.write(frame)

            if self.verbose:
                logging.info("Writing frame.")

        out.release()
        cv2.destroyAllWindows()

        if self.verbose:
            logging.info(f"Video saved to {self.video_path}")
        subprocess.run(["open", self.video_path])
