# import torch
import matplotlib.pyplot as plt
from reflect.examples.test_env import TestEnvironment
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.observation_model import ObservationalModel


if __name__ == "__main__":
    model = ObservationalModel(
        num_classes=32,
        num_latent=32
    )
    env = TestEnvironment()
    loader = EnvDataLoader(
        num_time_steps=4,
        batch_size=12,
        num_runs=100,
        rollout_length=5*5,
        transforms=lambda _: _,
        img_shape=(3, 64, 64),
        env=env,
        observation_model=model
    )
    loader.perform_rollout()
    fig, axs = plt.subplots(ncols=5, nrows=5)
    for i, img in enumerate(loader.img_buffer[0]):
        axs[i//5, i%5].imshow(img.permute(1, 2, 0))
    plt.show()