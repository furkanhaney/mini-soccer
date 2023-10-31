import pygame
import numpy as np
from state import PLAYER_RADIUS, BALL_RADIUS


class PygameRender:
    def __init__(self, resolution):
        self.resolution = resolution
        self.screen = pygame.display.set_mode(resolution, vsync=1)

    def decompose_state(self, state):
        player_positions = state.player_positions
        ball_position = state.ball_position
        return player_positions, ball_position

    def draw_borders(self):
        rect_borders = [
            (50, 50, self.resolution[0] - 50, 10),
            (50, 50, 10, self.resolution[1] - 50),
            (50, self.resolution[1] - 50, self.resolution[0] - 50, 10),
            (self.resolution[0] - 50, 50, 10, self.resolution[1] - 50),
        ]
        for rbd in rect_borders:
            pygame.draw.rect(self.screen, (255, 255, 255), rbd)

        # Draw goalposts
        goalpost_height = int(self.resolution[1] * 0.25)
        goalpost_width = int(self.resolution[0] * 0.20)

        # Inner rectangle size
        inner_goalpost_width = goalpost_width - 10
        inner_goalpost_height = goalpost_height - 10

        # Left goalpost (Outer White, Inner Green)
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (
                0,
                (self.resolution[1] - goalpost_height) // 2,
                goalpost_width,
                goalpost_height,
            ),
        )
        pygame.draw.rect(
            self.screen,
            (0, 128, 0),
            (
                5,
                ((self.resolution[1] - goalpost_height) // 2) + 5,
                inner_goalpost_width,
                inner_goalpost_height,
            ),
        )

        # Right goalpost (Outer White, Inner Green)
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (
                self.resolution[0] - goalpost_width,
                (self.resolution[1] - goalpost_height) // 2,
                goalpost_width,
                goalpost_height,
            ),
        )
        pygame.draw.rect(
            self.screen,
            (0, 128, 0),
            (
                self.resolution[0] - goalpost_width + 5,
                ((self.resolution[1] - goalpost_height) // 2) + 5,
                inner_goalpost_width,
                inner_goalpost_height,
            ),
        )

    def draw_frame(self, state):
        player_positions, ball_position = self.decompose_state(state)

        # Fill the background with green
        self.screen.fill((0, 128, 0))

        # Draw the borders and goalposts
        self.draw_borders()

        # Draw the center circle
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (self.resolution[0] // 2, self.resolution[1] // 2),
            100,
            5,
        )

        # Calculate the scaling factors for x and y axis
        scale_x, scale_y = self.resolution[0], self.resolution[1]

        # Draw players and ball
        for idx, position in enumerate(player_positions):
            x, y = (position * np.array([scale_x, scale_y])).astype(int)
            color = (255, 0, 0) if idx == 0 else (0, 0, 255)
            pygame.draw.circle(self.screen, color, (x, y), int(PLAYER_RADIUS * scale_x))

        ball_x, ball_y = (ball_position * np.array([scale_x, scale_y])).astype(int)
        pygame.draw.circle(
            self.screen, (255, 255, 255), (ball_x, ball_y), int(BALL_RADIUS * scale_x)
        )

    def render(self, state):
        for state in [state]:
            self.screen.fill((0, 0, 0))
            self.draw_frame(state)
            pygame.display.update()
