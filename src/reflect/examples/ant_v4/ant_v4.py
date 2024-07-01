from reflect.data.simple_rl_env import SimpleRLEnvironment
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.environment import Environment
from reflect.models.world_model import WorldModel
from reflect.utils import CSVLogger
import matplotlib.pyplot as plt
import torch
import click
import numpy as np
import pygame
import time
from reflect.examples.ant_v4.models import make_models

import gymnasium as gym
import click
import pygame
import time
import numpy as np



@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


@cli.command()
def play_real():

    pygame.init()
    HEIGHT, WIDTH = 256, 256
    display = pygame.display.set_mode((HEIGHT, WIDTH))

    black=(0,0,0)
    myFont = pygame.font.SysFont("Times New Roman", 18)
    randNumLabel = myFont.render("reward:", 1, black)

    env = gym.make(
        "Ant-v4",
        render_mode="rgb_array"
    )
    env.reset()
    screen = env.render()
    
    running = True
    reward_sum = 0
    while running:
        time.sleep(0.1)
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        # action = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        # action = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=np.float32)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = np.array([0, 1, 0, 0, 0, 0, 0, 0,], dtype=np.float32)
                if event.key == pygame.K_RIGHT:
                    action = np.array([0, -1, 0, 0, 0, 0, 0, 0,], dtype=np.float32)

        _, r, *_ = env.step(action)
        reward_sum += r
        diceDisplay = myFont.render(str(reward_sum), 1, black)
        screen = env.render()
        surface = pygame.surfarray.make_surface(screen)
        surface = pygame.transform.scale(surface, (HEIGHT, WIDTH))
        surface = pygame.transform.rotate(surface, -90)
        display.blit(surface, (0, 0))
        display.blit(randNumLabel, (0, 0))
        display.blit(diceDisplay, (60, 0))
        pygame.display.update()

    pygame.quit()
    env.close()

@cli.command()
def play_model():

    pygame.init()
    HEIGHT, WIDTH = 256, 256
    display = pygame.display.set_mode((HEIGHT, WIDTH))

    black=(0,0,0)
    myFont = pygame.font.SysFont("Times New Roman", 18)
    randNumLabel = myFont.render("reward:", 1, black)

    wm_env = make_models()
    wm_env.world_model.eval()
    wm_env.reset()
    running = True
    reward_sum = 0
    while running:
        time.sleep(0.1)
        action = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        # action = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
        # action = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1], dtype=torch.float32)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0,], dtype=torch.float32)
                if event.key == pygame.K_RIGHT:
                    action = torch.tensor([0, -1, 0, 0, 0, 0, 0, 0,], dtype=torch.float32)

        z_screen, r, d = wm_env.step_filter(action[None, None, :])
        reward_sum += r.item()
        diceDisplay = myFont.render(str(reward_sum), 1, black)
        screen = wm_env.world_model.decode(z_screen)
        screen = screen[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
        screen = (screen * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(screen)
        surface = pygame.transform.scale(surface, (HEIGHT, WIDTH))
        surface = pygame.transform.rotate(surface, -90)
        display.blit(surface, (0, 0))
        display.blit(randNumLabel, (0, 0))
        display.blit(diceDisplay, (60, 0))
        pygame.display.update()

    pygame.quit()


@cli.command()
def download_model():
    pass


if __name__ == "__main__":
    cli()