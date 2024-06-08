from reflect.data.simple_rl_env import SimpleRLEnvironment
from reflect.models.rl.td3_trainer import TD3Agent
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.environment import Environment
from reflect.models.world_model import WorldModel, WorldModelTrainingParams
from reflect.utils import CSVLogger
import matplotlib.pyplot as plt
import torch
import os
import click
import numpy as np
import pygame
from reflect.utils import create_z_dist
import time
from reflect.examples.simple_world.models import make_models

# RL PARAMETERS
ACTION_REG_SIG=0.05
ACTION_REG_CLIP = 0.2
ACTOR_UPDATE_FREQ = 2
TAU=5e-3
ACTOR_LR=1e-3
CRITIC_LR=1e-3
GAMMA=0.99
EPS=0.5

# WORLD MODEL PARAMETERS
BURNIN_STEPS=100
TRAIN_STEPS=1000
BATCH_SIZE=32
NUM_RUNS=1000
ROLLOUT_LENGTH=10

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
@click.option('--load', is_flag=True)
def train_full(load):
    observation_model, dynamic_model = make_models()

    params = WorldModelTrainingParams(
        reg_coeff=0.0,
        recon_coeff=1.0,
        dynamic_coeff=1.0,
        consistency_coeff=0.0,
        reward_coeff=10.0,
        done_coeff=0.0,
    )

    world_model = WorldModel(
        dynamic_model=dynamic_model,
        observation_model=observation_model,
        num_ts=t_dim-1,
        num_cat=num_cat,
        num_latent=latent_dim,
        params=params
    )

    if load:
        world_model.load(
            path='experiments/wm-td3/',
            name="world-model-checkpoint-trained.pth",
        )

    env = SimpleRLEnvironment(
        size=ENV_SIZE,
        num_threats=NUM_THREATS
    )

    agent = TD3Agent(
        state_dim=input_dim,
        action_space=env.action_space,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
    )

    agent.actor.bounds = torch.tensor([-2.0, 2.0])

    loader = EnvDataLoader(
        num_time_steps=t_dim,
        batch_size=12,
        num_runs=NUM_RUNS,
        rollout_length=5*5,
        transforms=lambda _: _,
        img_shape=(3, ENV_SIZE, ENV_SIZE),
        env=env,
        observation_model=world_model.observation_model,
    )

    rl_env = Environment(
        world_model=world_model,
        data_loader=loader,
        batch_size=64
    )

    os.makedirs("./experiments", exist_ok=True)
    os.makedirs(f"./experiments/wm-td3", exist_ok=True)

    logger = CSVLogger(
        path=f"./experiments/wm-td3/results.csv",
        fieldnames=[
            'recon_loss',
            'reg_loss',
            'consistency_loss',
            'dynamic_loss',
            'reward_loss',
            'done_loss',
            'rewards',
            'actor_loss',
            'critic_1_loss',
            'critic_2_loss'
        ]).initialize()

    for _ in range(BURNIN_STEPS):
        loader.perform_rollout()

    iteration = 0
    for i in range(100000):
        loader.perform_rollout()
        imgs, actions, rewards, dones = loader.sample()
        wm_history = world_model.update(
            o=imgs,
            a=actions,
            r=rewards,
            d=dones,
        )

        # train agent
        current_state, _ = rl_env.reset()
        rewards = []
        for _ in range(ROLLOUT_LENGTH):
            iteration += 1
            if rl_env.done:
                break
            current_state = current_state[rl_env.not_done]
            action = agent.actor.compute_action(
                current_state,
                eps=EPS
            )
            next_state, reward, done = rl_env.step(action)
            rl_agent_history = agent.update(
                state_samples=current_state,
                next_state_samples=next_state,
                action_samples=action,
                reward_samples=reward,
                done_samples=done
            )
            
            current_state = next_state

            rewards.append(reward.sum().item())

        reward_total_avg = sum(rewards)/BATCH_SIZE
        logger.log(
            **wm_history,
            **rl_agent_history,
            rewards=reward_total_avg
        )

        print((
            f"i: {i}, recon_loss: {wm_history['recon_loss']:.2f}, "
            f"'dynamic_loss': {wm_history['dynamic_loss']:.2f}, "
            f"'reward_loss': {wm_history['reward_loss']:.2f}, "
            f"'rewards': {reward_total_avg:.2f},"
        ))

        if (i > 0) and (i % 100) == 0:
            world_model.save("./experiments/wm-td3/")
            agent.save("./experiments/wm-td3/")
