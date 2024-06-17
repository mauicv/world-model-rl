# World Model RL

__Note__: This codebase is a work in progress.


This repo contains code to perform reinforcement learning inside learnt world models. My focus is n continuous control tasks. It's based on a number of different papers:

1. [World Models](https://arxiv.org/abs/1803.10122), see also [this blog post](https://worldmodels.github.io/). Introduce the idea of a world model. In particular they use a VAE to learn a latent space of the environment and an RNN to predict the next latent state. They then use a controller to generate actions based on the latent state. The controller is trained using CMA-ES.
2. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). Introduces the idea of training the actor to maximize the reward using the differentiability of the world model reward signal. Also extends this with a further value function for better reward estimation beyond the imagination rollouts.
3. [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193). This paper extends the original world models paper to discrete action spaces.
4. [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/abs/2303.07109). This paper extends the above to use a transformer instead of an RNN.

We implement bits and pieces from each. In particular we learn the discrete latent space using a CNN vae. We then learn the dynamics model using a transformer. Both these models are implemented [here](https://github.com/mauicv/transformers). When training the agent we train a value model to predict the rollout state value and then train the actor to maximize the value by directly propagating the gradients through the dynamic and value model.

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

### Simple World Model

This implements the full world model and RL loop on a very simple environment made up of a square grid of 16 squares. One of the squares is green and the agent (black square) must navigate to the goal while avoiding the red square. The agent can move up, down, left or right.

To see the options run:
    
```bash
python src/reflect/examples/simple_world/__init__.py --help
```