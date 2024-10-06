from reflect.data.simple_rl_env import SimpleRLEnvironment
from reflect.components.trainers.reward.reward_trainer import RewardGradTrainer
from reflect.components.models.actor import Actor
from reflect.data.loader import EnvDataLoader
from reflect.components.transformer_world_model.environment import Environment
from reflect.components.transformer_world_model import WorldModel
import torch
import click
import pygame
import time
from reflect.examples.simple_world.models import make_models
import numpy as np

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
def play_agent():

    pygame.init()
    HEIGHT, WIDTH = 256, 256
    display = pygame.display.set_mode((HEIGHT, WIDTH))

    black=(0,0,0)
    myFont = pygame.font.SysFont("Times New Roman", 18)
    randNumLabel = myFont.render("reward:", 1, black)

    observation_model, dynamic_model = make_models()

    world_model = WorldModel(
        dynamic_model=dynamic_model,
        observation_model=observation_model,
        num_ts=t_dim-1,
        num_cat=num_cat,
        num_latent=latent_dim,
    )

    world_model.load("./experiments/wm-td3/")

    env = SimpleRLEnvironment(
        size=ENV_SIZE,
        num_threats=NUM_THREATS
    )

    actor = Actor(
        input_dim=input_dim,
        action_space=env.action_space,
        hidden_dim=64,
        num_layers=2
    )

    agent = RewardGradTrainer(
        actor=actor,
        actor_lr=ACTOR_LR,
    )
    agent.load("./experiments/wm-td3/")
    agent.actor.bounds = torch.tensor([-2.0, 2.0])

    loader = EnvDataLoader(
        num_time_steps=t_dim,
        batch_size=12,
        num_runs=NUM_RUNS,
        rollout_length=5*5,
        transforms=lambda _: _,
        img_shape=(3, ENV_SIZE, ENV_SIZE),
        env=env,
        observation_model=world_model.observation_model
    )
    loader.perform_rollout()

    wm_env = Environment(
        world_model=world_model,
        data_loader=loader,
        batch_size=1,
        ignore_done=True
    )
    
    z_screen, _ = wm_env.reset()
    running = True
    reward_sum = 0
    while running:
        time.sleep(0.5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.actor.compute_action(z_screen)
        print(action)
        z_screen, r, *_ = wm_env.step_filter(action)
        reward_sum += r.item()
        diceDisplay = myFont.render(str(reward_sum), 1, black)
        screen = wm_env.world_model.decode(z_screen)
        screen = screen[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
        screen = (screen * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(screen)
        surface = pygame.transform.scale(surface, (HEIGHT, WIDTH))
        display.blit(surface, (0, 0))
        display.blit(randNumLabel, (0, 0))
        display.blit(diceDisplay, (60, 0))
        pygame.display.update()

    pygame.quit()