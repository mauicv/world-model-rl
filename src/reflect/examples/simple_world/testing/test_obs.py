from reflect.examples.simple_rl_env import SimpleRLEnvironment
from reflect.data.basic_loader import EnvDataLoader
from reflect.components.transformer_world_model import WorldModel
import matplotlib.pyplot as plt
import click
from reflect.utils import create_z_dist
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
def test_obs():
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

    o, *_ = loader.sample()
    o = o[0]
    world_model.observation_model.eval()
    r_o, _, logits = world_model.observation_model(o)
    dist = create_z_dist(logits)
    print(dist.base_dist.probs)

    fig, axs = plt.subplots(ncols=3, nrows=2)
    for i in range(3):
        axs[0, i].imshow(o[i].permute(1, 2, 0).detach())
        axs[1, i].imshow(r_o[i].permute(1, 2, 0).detach())
    plt.show()