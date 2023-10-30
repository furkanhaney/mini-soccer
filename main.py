from environment import Environment
from policy import RandomPolicy, KeyboardControlPolicy
from render import PygameRender, RenderMP4
import pygame
import time

if __name__ == "__main__":
    NUM_PLAYERS = 2
    ACTION_SPACE_DIM = 5

    pygame_renderer = PygameRender(resolution=(800, 800))

    policies = [KeyboardControlPolicy(ACTION_SPACE_DIM), RandomPolicy(ACTION_SPACE_DIM)]
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
