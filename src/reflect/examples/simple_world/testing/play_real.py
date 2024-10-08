from reflect.examples.simple_rl_env import SimpleRLEnvironment
import click
import numpy as np
import pygame
import time

# RL PARAMETERS
ACTION_REG_SIG=0.05
ACTION_REG_CLIP = 0.2
ACTOR_UPDATE_FREQ = 2
TAU=5e-3
ACTOR_LR=1e-3
CRITIC_LR=1e-3
GAMMA=0.99

# WORLD MODEL PARAMETERS
BURNIN_STEPS=100
TRAIN_STEPS=1000
BATCH_SIZE=32
NUM_RUNS=1000

NUM_THREATS=1
ENV_SIZE=4

hdn_dim=256
num_heads=4
latent_dim=8
num_cat=8
t_dim=3
input_dim=8*8
num_layers=4
dropout=0.05
a_size=2


@click.command()
def play_real():

    pygame.init()
    HEIGHT, WIDTH = 256, 256
    display = pygame.display.set_mode((HEIGHT, WIDTH))

    black=(0,0,0)
    myFont = pygame.font.SysFont("Times New Roman", 18)
    randNumLabel = myFont.render("reward:", 1, black)

    env = SimpleRLEnvironment(
        size=ENV_SIZE,
        num_threats=NUM_THREATS
    )
    env.reset()
    screen = env.render()
    
    running = True
    reward_sum = 0
    while running:
        time.sleep(0.1)
        action = np.array([0, 0], dtype=np.float32)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = np.array([-1, 0], dtype=np.float32)
                if event.key == pygame.K_RIGHT:
                    action = np.array([1, 0], dtype=np.float32)
                if event.key == pygame.K_UP:
                    action = np.array([0, -1], dtype=np.float32)
                if event.key == pygame.K_DOWN:
                    action = np.array([0, 1], dtype=np.float32)

        _, r, _ = env.step(action)
        reward_sum += r
        diceDisplay = myFont.render(str(reward_sum), 1, black)
        screen = env.render()
        surface = pygame.surfarray.make_surface(screen)
        surface = pygame.transform.scale(surface, (HEIGHT, WIDTH))
        display.blit(surface, (0, 0))
        display.blit(randNumLabel, (0, 0))
        display.blit(diceDisplay, (60, 0))
        pygame.display.update()

    pygame.quit()

