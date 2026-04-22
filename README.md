# World Model RL

World model Reinforcement Learning is a branch of RL that trains or uses models of the environment in order to solve the reinforcement learning task. This codebase contains implementations of some world model RL algorithms with a focus on continuous control.

![Imagined rollout for the Transformer based world model agent](/assets/tssm-imagined-rollout.gif)
![Real rollout for the Transformer based world model agent](/assets/tssm-real-rollout.gif)

## Examples:

**From state space:**

1. [~DCWM](https://colab.research.google.com/drive/1GIrvZrHiiOL3yP1ujiITtT9fn0JLJNtc?authuser=2#scrollTo=ktsRnu7lRAtU)
2. [TCRL](https://colab.research.google.com/drive/1zeSj1mC_IN3gqVDUFckVr8U0I3dbc_lI?authuser=2)
3. [TSSM](https://colab.research.google.com/drive/1UboktIA38uHxjH44houFtJQcb7eV9b3R)
4. [Discrete-RSSM](https://colab.research.google.com/drive/19rgE_bG905dnKVtN3kgjpikW8xOjt9nm)
5. [Continuous-RSSM](https://colab.research.google.com/drive/1GUJWQuB1JnR9WuvtG0YSv_m7Tv_7_QMe)

**From pixels:**

1. [TSSM](https://colab.research.google.com/drive/13SLo7x_sciK5QDZhKSeCgokpA4ZeLmRp)
2. [continuous-RSSM](https://colab.research.google.com/drive/1YVZw9tGJ9_YUnq11KAAE69ne3Zn-xIDH)
3. [discrete-RSSM](https://colab.research.google.com/drive/1_4exqY9vSCREyBbkJ9EPHoRMzuJOv0ke)


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

## Papers/References:


1. [World Models, David Ha, Jürgen Schmidhuber](https://arxiv.org/abs/1803.10122)
2. [DreamerV1, Dream to Control: Learning Behaviors by Latent Imagination, Danijar Hafner et al](https://arxiv.org/abs/1912.01603)
3. [DreamerV2, Mastering Atari with Discrete World Models, Danijar Hafner et al](https://arxiv.org/abs/2010.02193)
4. [Transformer-based World Models Are Happy With 100k Interactions, Jan Robine et al](https://arxiv.org/abs/2303.07109)
5. [TSSM, TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/abs/2202.09481)
7. [TCRL, Simplified Temporal Consistency Reinforcement Learning, Yi Zhao et al](https://arxiv.org/abs/2306.09466)
6. [DCWM, Discrete Codebook World Models for Continuous Control, Aidan Scannell et al](https://arxiv.org/abs/2503.00653)
