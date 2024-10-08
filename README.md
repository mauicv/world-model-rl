# World Model RL

This repo contains code to perform reinforcement learning inside learnt world models. My focus is n continuous control tasks. It's based on a number of different papers:

1. [World Models](https://arxiv.org/abs/1803.10122), see also [this blog post](https://worldmodels.github.io/). Introduce the idea of a world model. In particular they use a VAE to learn a latent space of the environment and an RNN to predict the next latent state. They then use a controller to generate actions based on the latent state. The controller is trained using CMA-ES.
2. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). Introduces the idea of training the actor to maximize the reward using the differentiability of the world model reward signal. Also extends this with a further value function for better reward estimation beyond the imagination rollouts.
3. [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193). This paper extends the original world models paper to discrete action spaces.
4. [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/abs/2303.07109). This paper extends the above to use a transformer instead of an RNN, however it uses actor critic methods to train the agent instead of differntiability of the world model reward signal.
5. [TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/abs/2202.09481) This paper also uses a transformer based architecture, albeit a different one to paper number 4. These authors do use the differentiability of the world model reward signal to train the model.

## Approach

The general approach taken in most of the above papers is to do three things:

1. Create a representation model that compresses the size of the input states (in our case images). We use a pair of encoder and decoder CNN networks for this. In particular we can choose to either a continuous or discrete latent representation. The benefit of compressing the state into a lower dimensional representation like this is that it makes training the world model less computationally expensive/easier.
2. Train a world model which takes the latent representation and predicts the next one in a sequence. I've implemented 2 approaches here. The recurrent state space model (RSSM, from papers 1, 2.) and the transformer state space model (TSSM, from papers 4, 5.).
3. Train an agent inside the world model to take actions that maximize rollout reward. In particular because we have a differentiable world model we can train the agent to do this directly rather than estimating the reward gradients using monte carlo methods. Note that we can obtain a world model rollout and just maximize the rewards directly but doing so means we're limited to the finite time horizon of the rollout. We can also train a value function that estimate the reward beyond this time horizon and doing so facilitates more stable learning.

### RSSM:

The RSSM world model uses a Recurrent neural network as the dynamic model (the model that predicts the next state). A limitation of this approach is that training requires an iteration step over the observed environment rollouts in order to calculate the hidden state at each step. The following example runs are taken from a model with limited training (~400 steps) on a google colab.

![Imagined rollout for the RNN based world model agent](/assets/rssm-imagined-rollout.gif)
![Real rollout for the RNN based world model agent](/assets/rssm-real-rollout.gif)


### TSSM:

The TSSM world model uses a transformer as the backend. Its implemented [here](https://github.com/mauicv/transformers) and based on [karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). The transformer acts on a sequence of states, rewards and actions. It embeds each as a separate token and and predicts the next state and reward in much the same way such models are applied to language modelling. We use relative positional embeddings introduced by Shaw et al in [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155v2).

Because the transformer model has no hidden state bottleneck in the same way the RNN does, it can be trained all at once instead of requiring iterating through the real environment rollout. The following are generated and real rollouts from an agent trained via the TSSM world model - again trained in a google collab but for longer.

![Imagined rollout for the Transformer based world model agent](/assets/tssm-imagined-rollout.gif)
![Real rollout for the Transformer based world model agent](/assets/tssm-real-rollout.gif)


### Note:

The above are not fair comparisons because they are trained for different periods of time - but my experience was that the transformer world model trains slower than the RNN based world model - took about double the amount of time to get to comparable performance. There are a couple of reasons this might be the case.

1. The transformer world model reward target has many more gradients paths through the model than the recurrent model. This potentially leads to instability when training the agent. This is the argument put forward in [Do Transformer World Models Give Better Policy Gradients?](https://arxiv.org/abs/2402.05290) by Ma et al.
2. The transformer model is a bigger model than the RSSM and the size of the model comes with trade offs - in this case shorter rollouts. The RSSM is trained on longer (15 steps) rollouts than the transformer model is (10 steps).


## Colab Examples:

1. [TSSM colab Example](https://colab.research.google.com/drive/1VgJ7E-THAOO1kPk7UWgfpbI0kNF_6gTi)
2. [RSSM-continuous colab Example](https://colab.research.google.com/drive/1Lj1Bhg5vwQJAhS_Ehq5X_w6AF3MTxfnk)
2. [RSSM-discrete colab Example](https://colab.research.google.com/drive/1PBCTxeD_x_aKmc0ciNfpzB_8E4zkIIEo)

## Whats next:

There are a couple of avenues for other RL projects here. 

1. In [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/abs/2206.14176) Wu et al apply the RSSM world model to different real world robots.
2. Deng et al utilize [structured state space models](https://arxiv.org/abs/2111.00396) as the dynamic model in [Facing Off World Model Backbones: RNNs, Transformers, and S4](https://arxiv.org/abs/2307.02064).

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
