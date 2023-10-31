from environment import Environment
from policy import RandomPolicy, FollowPolicy, KeyboardControlPolicy, LinearPolicy
from render import PygameRender
from render_mp4 import RenderMP4
import pygame
import time
from q_learning import QPolicy, DeepQLearning

if __name__ == "__main__":
    NUM_PLAYERS = 2
    ACTION_SPACE_DIM = 2
    STATE_DIM = 5 + NUM_PLAYERS * 4

    pygame_renderer = PygameRender(resolution=(1280, 720))

    model = DeepQLearning(STATE_DIM, hidden_dim=16)
    # player1 = FollowPolicy(action_space_dim=ACTION_SPACE_DIM, player_num=0)
    # player1 = KeyboardControlPolicy(ACTION_SPACE_DIM)
    player1 = QPolicy(
        ACTION_SPACE_DIM, STATE_DIM, player_num=0, verbose=True, model=model
    )
    player2 = QPolicy(
        ACTION_SPACE_DIM, STATE_DIM, player_num=1, verbose=False, model=model
    )
    policies = [player1, player2]
    env = Environment(num_players=NUM_PLAYERS, policies=policies, tick_rate=60)

    start_time = time.time()
    update_time = time.time()
    run_game_time = 30
    end_time = pygame.time.get_ticks() + run_game_time * 1000

    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        update_time = time.time()
        pygame_renderer.render(env.state)
        env.tick(update_time - start_time)
        start_time = update_time

    print("Finished the game.")
    pygame.quit()

    video_renderer = RenderMP4(
        env.state_history,
        resolution=(1280, 720),
        framerate=60,
        video_path="env_000.mp4",
        verbose=False,
    )
    env.render(video_renderer)
