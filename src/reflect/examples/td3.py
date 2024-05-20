from reflect.models.td3_policy.trainer import Agent
from reflect.models.td3_policy.actor import Actor
from reflect.models.td3_policy.replay_buffer import ReplayBuffer
from reflect.utils import CSVLogger
import gymnasium as gym
import torch
import click
import os


ROLLOUT_LENGTH = 100
REPLAY_BUFFER_SIZE=1000000
BATCH_SIZE=128
ITERATIONS=10000
EPS=0.5
BURNIN_EPOCHS=100
ACTION_REG_SIG=0.05
ACTION_REG_CLIP = 0.2
ACTOR_UPDATE_FREQ = 2
TAU=5e-3
ACTOR_LR=1e-3
CRITIC_LR=1e-3
GAMMA=0.99


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


@click.option('--environment', default='BipedalWalker-v3')
@cli.command()
def plot(environment):
    click.echo(f"Plotting {environment}")
    logger = CSVLogger(
        path=f"./experiments/{environment}/results.csv",
        fieldnames=[
            'true_reward',
            'target_rewards',
            'actor_loss',
            'critic_1_loss',
            'critic_2_loss'
        ])

    logger.plot(
        [
            ("true_reward", "target_rewards"),
            ("critic_1_loss", "critic_2_loss"),
            ("actor_loss",)
        ]
    )


@click.option('--environment', default='BipedalWalker-v3')
@cli.command()
def play(environment):
    click.echo(f"Playing {environment}")
    env = gym.make(environment, render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space
    actor = Actor(
        input_dim=state_dim,
        action_space=action_space,
    )
    models_state_dict = torch.load(f"./experiments/{environment}/models.pth")
    actor.load_state_dict(models_state_dict['actor'])
    actor.eval()
    state, _ = env.reset()
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action = actor(state).detach().numpy()
        next_state, reward, done, *_ = env.step(action)
        env.render()
        state = next_state
    env.close()



@click.option('--environment', default='BipedalWalker-v3')
@cli.command()
def train(environment):
    click.echo('Syncing')

    print("----------------------------------------")
    print("ROLLOUT_LENGTH\t\t", ROLLOUT_LENGTH)
    print("REPLAY_BUFFER_SIZE\t", REPLAY_BUFFER_SIZE)
    print("BATCH_SIZE\t\t", BATCH_SIZE)
    print("ITERATIONS\t\t", ITERATIONS)
    print("EPS\t\t\t", EPS)
    print("BURNIN_EPOCHS\t\t", BURNIN_EPOCHS)
    print("ACTION_REG_SIG\t\t", ACTION_REG_SIG)
    print("ACTION_REG_CLIP\t\t", ACTION_REG_CLIP)
    print("ACTOR_UPDATE_FREQ\t", ACTOR_UPDATE_FREQ)
    print("TAU\t\t\t", TAU)
    print("ACTOR_LR\t\t", ACTOR_LR)
    print("CRITIC_LR\t\t", CRITIC_LR)
    print("GAMMA\t\t\t", GAMMA)
    print("----------------------------------------")
    print("")

    env = gym.make(environment)

    agent = Agent(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
    )

    replay_buffer = ReplayBuffer(
        action_space_dim=env.action_space.shape[0],
        state_space_dim=env.observation_space.shape[0],
        size=REPLAY_BUFFER_SIZE,
        sample_size=BATCH_SIZE
    )

    os.makedirs("./experiments", exist_ok=True)
    os.makedirs(f"./experiments/{environment}", exist_ok=True)

    logger = CSVLogger(
        path=f"./experiments/{environment}/results.csv",
        fieldnames=[
            'true_reward',
            'target_rewards',
            'actor_loss',
            'critic_1_loss',
            'critic_2_loss'
        ]).initialize()

    def test_policy(
            env,
            policy,
            rollout_length=ROLLOUT_LENGTH
        ):
        state, _ = env.reset()
        action = policy.compute_action(state, eps=0)[0]
        rewards = []
        actions = []
        for _ in range(rollout_length):
            next_state, reward, done, _, _ = env.step(action.numpy())
            rewards.append(reward)
            state = next_state
            action = policy.compute_action(state, eps=0)[0]
            actions.append(action)
            if done: break
        return sum(rewards), sum(actions)/len(actions)


    rewards, _ = test_policy(env, agent.actor, rollout_length=ROLLOUT_LENGTH)
    print(f"Initial rewards: {rewards}")


    for iterations in range(ITERATIONS):
        actor_losses = []
        critic_1_losses = []
        critic_2_losses = []
        current_state, _ = env.reset()
        action = agent.actor.compute_action(current_state)[0]

        for _ in range(ROLLOUT_LENGTH):
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push((current_state, next_state, action, reward, done))
            if done: break
            current_state = next_state
            action = agent.actor.compute_action(
                current_state,
                eps=EPS
            )[0]

            if replay_buffer.ready and (iterations > BURNIN_EPOCHS):
                state_samples, next_state_samples, action_samples, \
                    reward_samples, done_samples = replay_buffer.sample()

                state_samples = torch.tensor(state_samples)
                next_state_samples = torch.tensor(next_state_samples)
                reward_samples = torch.tensor(reward_samples)
                done_samples = torch.tensor(done_samples)
                action_samples = torch.tensor(action_samples)
                nosie = torch.randn_like(action_samples) * ACTION_REG_SIG
                action_samples = action_samples + torch.clamp(
                    nosie,
                    -ACTION_REG_CLIP,
                    ACTION_REG_CLIP
                )

                q_loss_1, q_loss_2 = agent.update_critic(
                    state_samples,
                    next_state_samples,
                    action_samples,
                    reward_samples,
                    done_samples,
                    gamma=GAMMA
                )
                critic_1_losses.append(q_loss_1)
                critic_2_losses.append(q_loss_2)

                if (iterations % ACTOR_UPDATE_FREQ) == 0:
                    actor_loss = agent.update_actor(state_samples)
                    actor_losses.append(actor_loss)
                    agent.update_actor_target_network()

                agent.update_critic_target_network()

        if (iterations > BURNIN_EPOCHS) and (iterations % ACTOR_UPDATE_FREQ == 0):
            target_rewards, avg_action = test_policy(
                env=env,
                policy=agent.target_actor,
                rollout_length=ROLLOUT_LENGTH
            )

            true_rewards, avg_action = test_policy(
                env=env,
                policy=agent.actor,
                rollout_length=ROLLOUT_LENGTH
            )

            data = {
                'true_reward': true_rewards,
                'target_rewards': target_rewards,
                'actor_loss': sum(actor_losses)/len(actor_losses),
                'critic_1_loss': sum(critic_1_losses)/len(critic_1_losses),
                'critic_2_loss': sum(critic_2_losses)/len(critic_2_losses)
            }
            logger.log(**data)

            print(f"Iterations: {iterations}, True rewards: {true_rewards:.2f}, Target rewards: {target_rewards:.2f}")
        
        if (iterations > 0) and (iterations % 10) == 0:
            agent.save(f"./experiments/{environment}/models.pth")

if __name__ == "__main__":
    cli()