from reflect.examples.test_env import SimpleEnvironment
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.observation_model import ObservationalModel, LatentSpace
from reflect.models.world_model.environment import Environment
from reflect.models.world_model import WorldModel
from pytfex.convolutional.decoder import DecoderLayer, Decoder
from pytfex.convolutional.encoder import EncoderLayer, Encoder
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.models.world_model.head import Head
from reflect.models.world_model.embedder import Embedder
from reflect.utils import CSVLogger
import matplotlib.pyplot as plt
import torch
import os
import click
import numpy as np
import pygame
from reflect.utils import create_z_dist
import time

BURNIN_STEPS=100
TRAIN_STEPS=1000
BATCH_SIZE=32
NUM_RUNS=1000
env_size=16
hdn_dim=128
num_heads=2
latent_dim=4
num_cat=4
input_dim=latent_dim*num_cat
t_dim=3
num_layers=4
dropout=0.0
a_size=2


def make_models():
    dynamic_model = GPT(
        dropout=dropout,
        hidden_dim=hdn_dim,
        num_heads=num_heads,
        embedder=Embedder(
            z_dim=input_dim,
            a_size=a_size,
            hidden_dim=hdn_dim
        ),
        head=Head(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hdn_dim
        ),
        layers=[
            TransformerLayer(
                hidden_dim=hdn_dim,
                attn=RelativeAttention(
                    hidden_dim=hdn_dim,
                    num_heads=num_heads,
                    num_positions=t_dim,
                    dropout=dropout
                ),
                mlp=MLP(
                    hidden_dim=hdn_dim,
                    dropout=dropout
                )
            ) for _ in range(num_layers)
        ]
    )

    encoder_layers = [
        EncoderLayer(
            in_channels=64,
            out_channels=128,
            num_residual=0,
        ),
        EncoderLayer(
            in_channels=128,
            out_channels=256,
            num_residual=0,
        ),
    ]

    encoder = Encoder(
        nc=3,
        ndf=64,
        layers=encoder_layers,
    )
    
    decoder_layers = [
        DecoderLayer(
            in_filters=256,
            out_filters=128,
            num_residual=0,
        ),
        DecoderLayer(
            in_filters=128,
            out_filters=64,
            num_residual=0,
        ),
    ]

    decoder = Decoder(
        nc=3,
        ndf=64,
        layers=decoder_layers,
        output_activation=torch.nn.Sigmoid(),
    )

    latent_space = LatentSpace(
        num_latent=latent_dim,
        num_classes=num_cat,
        input_shape=(256, 4, 4),
    )

    observation_model = ObservationalModel(
        encoder=encoder,
        decoder=decoder,
        latent_space=latent_space,
    )

    return observation_model, dynamic_model


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


@cli.command()
def plot():
    logger = CSVLogger(path=f"./experiments/wm/results.csv",
        fieldnames=[
            'recon_loss',
            'reg_loss',
            'consistency_loss',
            'dynamic_loss',
            'reward_loss',
            'done_loss'
        ])

    logger.plot(
        [
            ('recon_loss',),
            # ('reg_loss',),
            ('consistency_loss',),
            ('dynamic_loss',),
            ('reward_loss',),
            # ('done_loss', )
        ]
    )
    

@cli.command()
def train():
    observation_model, dynamic_model = make_models()

    world_model = WorldModel(
        dynamic_model=dynamic_model,
        observation_model=observation_model,
        num_ts=t_dim-1,
        num_cat=num_cat,
        num_latent=latent_dim,
    )

    env = SimpleEnvironment(
        size=env_size
    )

    loader = EnvDataLoader(
        num_time_steps=t_dim,
        batch_size=12,
        num_runs=NUM_RUNS,
        rollout_length=5*5,
        transforms=lambda _: _,
        img_shape=(3, env_size, env_size),
        env=env,
        observation_model=world_model.observation_model
    )

    os.makedirs("./experiments", exist_ok=True)
    os.makedirs(f"./experiments/wm", exist_ok=True)

    logger = CSVLogger(
        path=f"./experiments/wm/results.csv",
        fieldnames=[
            'recon_loss',
            'reg_loss',
            'consistency_loss',
            'dynamic_loss',
            'reward_loss',
            'done_loss'
        ]).initialize()

    for _ in range(BURNIN_STEPS):
        loader.perform_rollout()

    for i in range(10000):
        loader.perform_rollout()
        imgs, actions, rewards, dones = loader.sample()
        history = world_model.update(
            o=imgs,
            a=actions,
            r=rewards,
            d=dones,
        )
        logger.log(**history)
        print(f"i: {i}, recon_loss: {history['recon_loss']:.2f}, 'dynamic_loss': {history['dynamic_loss']:.2f}")

        if (i > 0) and (i % 100) == 0:
            world_model.save("./experiments/wm/")


@cli.command()
def play_real():

    pygame.init()
    HEIGHT, WIDTH = 256, 256
    display = pygame.display.set_mode((HEIGHT, WIDTH))

    env = SimpleEnvironment(
        size=env_size
    )
    env.reset()
    screen = env.render()
    
    running = True
    while running:
        time.sleep(0.1)
        action = np.array([0, 0], dtype=np.float32)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = np.array([-0.8, 0], dtype=np.float32)
                if event.key == pygame.K_RIGHT:
                    action = np.array([0.8, 0], dtype=np.float32)
                if event.key == pygame.K_UP:
                    action = np.array([0, -0.8], dtype=np.float32)
                if event.key == pygame.K_DOWN:
                    action = np.array([0, 0.8], dtype=np.float32)
        # action = env.action_space.sample()
        env.step(action)
        screen = env.render()
        surface = pygame.surfarray.make_surface(screen)
        surface = pygame.transform.scale(surface, (HEIGHT, WIDTH))
        display.blit(surface, (0, 0))
        pygame.display.update()

    pygame.quit()


@cli.command()
def play_model():

    pygame.init()
    HEIGHT, WIDTH = 256, 256
    display = pygame.display.set_mode((HEIGHT, WIDTH))

    observation_model, dynamic_model = make_models()

    world_model = WorldModel(
        dynamic_model=dynamic_model,
        observation_model=observation_model,
        num_ts=t_dim-1,
        num_cat=num_cat,
        num_latent=latent_dim,
    )

    world_model.load("./experiments/wm/")

    env = SimpleEnvironment(
        size=env_size
    )

    loader = EnvDataLoader(
        num_time_steps=t_dim,
        batch_size=12,
        num_runs=NUM_RUNS,
        rollout_length=5*5,
        transforms=lambda _: _,
        img_shape=(3, env_size, env_size),
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
    
    wm_env.reset()
    running = True
    while running:
        time.sleep(0.1)

        action = torch.tensor([0, 0], dtype=torch.float32)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = torch.tensor([-1, 0], dtype=torch.float32)
                if event.key == pygame.K_RIGHT:
                    action = torch.tensor([1, 0], dtype=torch.float32)
                if event.key == pygame.K_UP:
                    action = torch.tensor([0, -1], dtype=torch.float32)
                if event.key == pygame.K_DOWN:
                    action = torch.tensor([0, 1], dtype=torch.float32)

        z_screen, *_ = wm_env.step(action[None, None, :])
        screen = wm_env.world_model.decode(z_screen)
        screen = screen[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
        screen = (screen * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(screen)
        surface = pygame.transform.scale(surface, (HEIGHT, WIDTH))
        display.blit(surface, (0, 0))
        pygame.display.update()

    pygame.quit()

@cli.command()
def test_observation_model():
    observation_model, dynamic_model = make_models()

    world_model = WorldModel(
        dynamic_model=dynamic_model,
        observation_model=observation_model,
        num_ts=t_dim-1,
        num_cat=num_cat,
        num_latent=latent_dim,
    )

    world_model.load("./experiments/wm/")

    env = SimpleEnvironment(
        size=env_size
    )

    loader = EnvDataLoader(
        num_time_steps=t_dim,
        batch_size=12,
        num_runs=NUM_RUNS,
        rollout_length=5*5,
        transforms=lambda _: _,
        img_shape=(3, env_size, env_size),
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


if __name__ == "__main__":
    cli()

