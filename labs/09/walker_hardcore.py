#!/usr/bin/env python3
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch
import random

import copy

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate_actor", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_critic", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")

USE_WANDB = True

if USE_WANDB:
    import wandb

class Actor(torch.nn.Module):
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(args.hidden_layer_size, env.action_space.shape[0])
        self.tanh = torch.nn.Tanh()
        self.scaling_factor = env.action_space.high[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.tanh(x)
        x = x * self.scaling_factor
        return x

class Critic(torch.nn.Module):
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], args.hidden_layer_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(args.hidden_layer_size, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], dim=1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create:
        # - an actor, which starts with states and returns actions.
        #   Usually, one or two hidden layers are employed. As in the
        #   paac_continuous, to keep the actions in the required range, you
        #   should apply properly scaled `torch.tanh` activation.
        #
        # - a target actor as the copy of the actor using `copy.deepcopy`.
        #
        # - a critic, starting with given states and actions, producing predicted
        #   returns. The states and actions are usually concatenated and fed through
        #   two more hidden layers, before computing the returns with the last output layer.
        #
        # - a target critic as the copy of the critic using `copy.deepcopy`.
        #

        self.actor = Actor(env, args).to(self.device)
        self.actor.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate_actor)
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = Critic(env, args).to(self.device)
        self.critic.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.learning_rate_critic)
        self.critic_loss = torch.nn.MSELoss()
        self.target_critic = copy.deepcopy(self.critic)

        self.target_tau = args.target_tau


    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Separately train:
        # - the actor using the DPG loss,
        # - the critic using MSE loss.
        #
        # Furthermore, update the target actor and critic networks by
        # exponential moving average with weight `args.target_tau`. A possible
        # implementation is the following:
        #   for param, target_param in zip(source.parameters(), target.parameters()):
        #       target_param.data.mul_(1 - target_tau)
        #       target_param.data.add_(target_tau * param.data)

        self.actor.train()
        self.critic.train()

        self.actor_optimizer.zero_grad()
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_pred = self.critic(states, actions)
        critic_loss = self.critic_loss(critic_pred, returns).mean()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.mul_(1 - self.target_tau)
            target_param.data.add_(self.target_tau * param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.mul_(1 - self.target_tau)
            target_param.data.add_(self.target_tau * param.data)


    def save_to_cpu(self, name):
        self.actor.cpu()
        self.target_actor.cpu()
        self.critic.cpu()
        self.target_critic.cpu()
        torch.save(self.actor.state_dict(), f"actor_{name}.pth")
        torch.save(self.target_actor.state_dict(), f"target_actor_{name}.pth")
        torch.save(self.critic.state_dict(), f"critic_{name}.pth")
        torch.save(self.target_critic.state_dict(), f"target_critic_{name}.pth")
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

    def load_from_cpu(self, name):
        self.actor.load_state_dict(torch.load(f"actor_{name}.pth"))
        self.target_actor.load_state_dict(torch.load(f"target_actor_{name}.pth"))
        self.critic.load_state_dict(torch.load(f"critic_{name}.pth"))
        self.target_critic.load_state_dict(torch.load(f"target_critic_{name}.pth"))
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted actions by the actor.
        with torch.no_grad():
            actions = self.actor(states)
            return actions

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted returns -- predict actions by the target actor
        # and evaluate them using the target critic.
        with torch.no_grad():
            actions = self.target_actor(states)
            values = self.target_critic(states, actions)
            return values


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; the `max_length` parameter can be passed to limit its size.
    replay_buffer = wrappers.ReplayBuffer()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # TODO: Predict the action by calling `network.predict_actions`.
            action = network.predict_actions(np.array([state]))[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == -100:
                reward = 0
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = not args.recodex
    env_thresholds = {"Pendulum-v1": -180, "InvertedDoublePendulum-v5" : 9200, "BipedalWalker-v3": 210, "BipedalWalkerHardcore-v3": 110}
    episode = 0
    best = 100
    random_value = random.randint(0, 1000000)
    random_value = 514009
    wandb.log({"random_value": random_value})
   # network.load_from_cpu(f"{args.env}_{random_value}")
    if USE_WANDB:
        wandb.log({"episode": episode})
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset()[0], False
            noise.reset()
            while not done:
                # TODO: Predict actions by calling `network.predict_actions`
                # and adding the Ornstein-Uhlenbeck noise. As in paac_continuous,
                # clip the actions to the `env.action_space.{low,high}` range.
                action = network.predict_actions(np.array([state]))[0] + noise.sample()
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if done:
                    reward = 0

                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) < 4 * args.batch_size:
                    continue
                batch = replay_buffer.sample(args.batch_size, np.random)
                states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
                # TODO: Perform the training

                rewards = rewards[:,np.newaxis]
                returns = rewards + args.gamma * (~dones)[:, np.newaxis]*network.predict_values(next_states)
                network.train(states, actions, returns)

            episode += 1

            if USE_WANDB:
                wandb.log({"episode": episode})


        # Periodic evaluation
        returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
        print("Evaluation after episode {}: {:.2f}".format(env.episode, np.mean(returns)))

        if USE_WANDB:
            wandb.log({"return": np.mean(returns)})

        if np.mean(returns) > best:
            best = np.mean(returns)
            print(f"Best so far: {best}")
            if USE_WANDB:
                wandb.log({"best": best})
            network.save_to_cpu(f"{args.env}_{random_value}")

    network.load_from_cpu(args.env)
    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if USE_WANDB:
        wandb.init(project=args.env)
        wandb.config.update(args)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
