# World Model RL

__Note__: This codebase is a work in progress.


This repo contains code to perform reinforcement learning inside learnt world models. My focus is n continuous control tasks. It's based on a number of different papers:

1. [World Models](https://arxiv.org/abs/1803.10122), see also [this blog post](https://worldmodels.github.io/). Introduce the idea of a world model. In particular they use a VAE to learn a latent space of the environment and an RNN to predict the next latent state. They then use a controller to generate actions based on the latent state. The controller is trained using CMA-ES.
2. [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193). This paper extends the original world models paper to discrete action spaces. They train the controller using policy gradients.
3. [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/abs/2303.07109). This paper extends the above to use a transformer instead of an RNN.
4. [Twin Delayed Deep Deterministic Policy Gradient](https://arxiv.org/abs/1802.09477). This paper extends [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) to use two Q networks and a target policy network. DDPG is an RL algorithm that is particularly well suited to continuous action spaces.

We implement bits and pieces from each. In particular we learn the discrete latent space using a CNN vae. We then learn the dynamics model using a transformer. Both these models are implemented [here](https://github.com/mauicv/transformers). We then train the controller using [Twin Delayed Deep Deterministic Policy Gradient](https://arxiv.org/abs/1802.09477). 

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Tests

```bash
source venv/bin/activate
pytest src
```

## Examples

Three examples are provided in the `examples` directory.

### Simple World Model

This implements the full world model and RL loop on a very simple environment made up of a square grid of 16 squares. One of the squares is green and the agent (black square) must navigate to the goal while avoiding the red square. The agent can move up, down, left or right.

To see the options run:
    
```bash
python src/reflect/examples/simple_world/__init__.py --help
```

### TD3 (Twin Delayed Deep Deterministic Policy Gradient) 

Based on [Fujimoto, et al](https://arxiv.org/abs/1802.09477)'s paper on TD3. To see the options run:

```bash
python src/reflect/examples/td3.py --help
```

### NES (Natural Evolution Strategies)

Based on [Wierstra, et al](https://arxiv.org/abs/1106.4487)'s paper on natural evolutionary strategies. Note that this isn't used in the final implementation of the world model although ES has been used successfully in this domain, see [Ha](https://worldmodels.github.io/).

```bash
python src/reflect/examples/nes.py
```
