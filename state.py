import numpy as np
from action import Actions

PLAYER_ACCELERATION = 1.5
MAX_PLAYER_SPEED = 0.5

PLAYER_RADIUS = 0.02
PLAYER_MASS = 5
PLAYER_FRICTION = 0.01

BALL_RADIUS = 0.01
BALL_MASS = 0.1
BALL_FRICTION = 0.1

ELASTICITY = 0.7


class State:
    def __init__(self, num_players: int):
        self.player_positions = np.zeros(((num_players, 2)))
        self.player_positions[0] = 0.3, 0.5
        self.player_positions[1] = 0.7, 0.5
        self.player_velocities = np.zeros((num_players, 2))
        self.ball_position = np.array([0.5, 0.5])
        self.ball_velocity = np.random.uniform(-1, 1, 2)
        self.num_players = num_players

    def limit_speed(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed > MAX_PLAYER_SPEED:
            return (velocity / speed) * MAX_PLAYER_SPEED
        return velocity

    def get_friction(self, velocity: np.ndarray, friction: float) -> np.ndarray:
        return velocity * friction

    def update_player_positions_and_velocities(
        self, player_actions: Actions, delta_time: float
    ):
        action_array = player_actions.actions

        # Apply acceleration
        self.player_velocities += action_array * delta_time * PLAYER_ACCELERATION

        # Apply friction
        self.player_velocities -= (
            self.get_friction(self.player_velocities, PLAYER_FRICTION) * delta_time
        )

        # Limit speed and update position
        self.player_velocities = np.apply_along_axis(
            self.limit_speed, 1, self.player_velocities
        )
        self.player_positions += self.player_velocities * delta_time

        # Clip positions and reset velocities if at borders.
        for i in range(self.num_players):
            if np.any(self.player_positions[i] == [0.05, 0.05]) or np.any(
                self.player_positions[i] == [0.95, 0.95]
            ):
                self.player_velocities[i] = np.zeros(2)
        self.player_positions = np.clip(self.player_positions, 0.05, 0.95)

    def resolve_player_collisions(self):
        for i in range(self.num_players):
            for j in range(i + 1, self.num_players):
                distance = np.linalg.norm(
                    self.player_positions[i] - self.player_positions[j]
                )
                if distance < 2 * PLAYER_RADIUS:
                    overlap = 2 * PLAYER_RADIUS - distance
                    normal = (self.player_positions[i] - self.player_positions[j]) / (
                        distance + 1e-7
                    )

                    self.player_positions[i] += (overlap / 2) * normal
                    self.player_positions[j] -= (overlap / 2) * normal

    def resolve_collisions(self, elasticity: float):
        for i in range(self.num_players):
            player_pos = self.player_positions[i]
            distance_to_ball = np.linalg.norm(player_pos - self.ball_position)

            if distance_to_ball < (BALL_RADIUS + PLAYER_RADIUS):
                overlap = BALL_RADIUS + PLAYER_RADIUS - distance_to_ball
                normal = (player_pos - self.ball_position) / distance_to_ball

                # Move the positions to resolve the overlap
                self.player_positions[i] += (overlap / 2) * normal
                self.ball_position -= (overlap / 2) * normal

                # Calculate velocities along the normal direction for both ball and player
                ball_speed_normal = np.dot(self.ball_velocity, normal)
                player_speed_normal = np.dot(self.player_velocities[i], normal)

                # Calculate new speeds along normal after collision using mass
                total_mass = PLAYER_MASS + BALL_MASS
                new_ball_speed_normal = (
                    (2 * PLAYER_MASS / total_mass) * player_speed_normal
                ) + ((1 - 2 * PLAYER_MASS / total_mass) * ball_speed_normal)
                new_player_speed_normal = (
                    (2 * BALL_MASS / total_mass) * ball_speed_normal
                ) + ((1 - 2 * BALL_MASS / total_mass) * player_speed_normal)

                # Apply the elasticity factor
                change_in_ball_speed_normal = new_ball_speed_normal - ball_speed_normal
                change_in_player_speed_normal = (
                    new_player_speed_normal - player_speed_normal
                )

                new_ball_speed_normal = (
                    ball_speed_normal + elasticity * change_in_ball_speed_normal
                )
                new_player_speed_normal = (
                    player_speed_normal + elasticity * change_in_player_speed_normal
                )

                # Update the velocities of the ball and player
                self.ball_velocity += (
                    new_ball_speed_normal - ball_speed_normal
                ) * normal
                self.player_velocities[i] += (
                    new_player_speed_normal - player_speed_normal
                ) * normal

    def update_ball_position(self, delta_time: float):
        # Apply friction to ball
        self.ball_velocity -= (
            self.get_friction(self.ball_velocity, BALL_FRICTION) * delta_time
        )
        self.ball_position += self.ball_velocity * delta_time

        # Detect and resolve collisions with walls
        for i in range(2):  # Loop over x and y axes
            if (
                self.ball_position[i] <= BALL_RADIUS
                or self.ball_position[i] >= 1 - BALL_RADIUS
            ):
                self.ball_velocity[i] = -self.ball_velocity[i]
                # Reposition the ball to avoid overlap with the wall
                self.ball_position[i] = np.clip(
                    self.ball_position[i], BALL_RADIUS, 1 - BALL_RADIUS
                )
        self.ball_position = np.clip(self.ball_position, BALL_RADIUS, 1 - BALL_RADIUS)

    def update(self, player_actions: Actions, delta_time: float):
        self.update_player_positions_and_velocities(player_actions, delta_time)
        self.resolve_collisions(ELASTICITY)
        self.resolve_player_collisions()
        self.update_ball_position(delta_time)

    def to_numpy(self, player=0):
        arr = np.concatenate(
            [
                np.ones(1) * player,
                self.player_positions.flatten(),
                self.player_velocities.flatten(),
                self.ball_position.flatten(),
                self.ball_velocity.flatten(),
            ]
        )
        return arr
