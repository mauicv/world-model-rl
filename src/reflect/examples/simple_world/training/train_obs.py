from reflect.examples.simple_rl_env import SimpleRLEnvironment
from reflect.data.basic_loader import EnvDataLoader
from reflect.components.transformer_world_model import WorldModel, WorldModelTrainingParams
from reflect.utils import CSVLogger
import os
import click
from reflect.examples.simple_world.models import make_models

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
def train_obs():
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

    env = SimpleRLEnvironment(
        size=ENV_SIZE,
        num_threats=NUM_THREATS
    )

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
        ]).initialize()

    for _ in range(BURNIN_STEPS):
        loader.perform_rollout()

    for i in range(100000):
        loader.perform_rollout()
        imgs, *_ = loader.sample()
        history = world_model.update_observation_model(o=imgs)

        logger.log(**history)
        print((
            f"i: {i}, recon_loss: {history['recon_loss']:.2f},"
            f"'reg_loss': {history['reg_loss']:.2f}"
        ))

        if (i > 0) and (i % 100) == 0:
            world_model.save("./experiments/wm-td3/")
