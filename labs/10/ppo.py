#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch

import multi_collect_environment
import wrappers

multi_collect_environment.register()

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="SingleCollect-v0", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.2, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=32, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=7, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=512, type=int, help="Steps for each worker to perform.")

import wandb

class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace) -> None:
        self._args = args

        # TODO: Create an actor using a single hidden layer with `args.hidden_layer_size`
        # units and ReLU activation, produce a policy with `action_space.n` discrete actions.
        self._actor = torch.nn.Sequential(
            self.init_layer(torch.nn.Linear(np.prod(observation_space.shape), args.hidden_layer_size)),
            torch.nn.ReLU(),
            self.init_layer(torch.nn.Linear(args.hidden_layer_size, action_space.n), std=0.01),
        ).to(self.device)

        # TODO: Create a critic (value predictor) consisting of a single hidden layer with
        # `args.hidden_layer_size` units and ReLU activation, and and output layer with a single output.
        self._critic = torch.nn.Sequential(
            self.init_layer(torch.nn.Linear(np.prod(observation_space.shape), args.hidden_layer_size)),
            torch.nn.ReLU(),
            self.init_layer(torch.nn.Linear(args.hidden_layer_size, 1), std=1.0)
        ).to(self.device)

        self._optimizer = torch.optim.Adam(list(self._actor.parameters()) + list(self._critic.parameters()), lr=args.learning_rate, eps=1e-5)

    def init_layer(self, layer, std=np.sqrt(2)):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, 0)
        return layer

    def save_actor(self, path: str):
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str):
        self._actor.load_state_dict(torch.load(path, map_location=self.device))

    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    def train(self, states: torch.Tensor, actions: torch.Tensor, action_probs: torch.Tensor,
              advantages: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform a single training step of the PPO algorithm.
        # For the policy model, the sum is the sum of:
        # - the PPO loss, where `self._args.clip_epsilon` is used to clip the probability ratio
        # - the entropy regularization with coefficient `self._args.entropy_regularization`.
        #   You can compute it for example using the `torch.distributions.Categorical` class.
        _, newlogprob, entropy = self.predict_actions(states, action=actions)
        newvalue = self.predict_values(states).squeeze()
        logratio = newlogprob - action_probs
        ratio = torch.exp(logratio)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = ratio * advantages
        pg_loss2 = torch.clamp(ratio, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon) * advantages
        ppo_loss = -torch.min(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = self._args.entropy_regularization * entropy.mean()

        loss = ppo_loss + v_loss - entropy_loss

        wandb.log({"ppo_loss": ppo_loss})
        wandb.log({"entropy_loss": entropy_loss})

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def predict_actions(self, states: torch.Tensor, greedy=False, action=None):
        logits = self._actor(states)
        probs = torch.distributions.categorical.Categorical(logits=logits)
        if action is None:
            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


    def predict_values(self, states: torch.Tensor):
        return self._critic(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Construct the network
    network = Network(env.observation_space, env.action_space, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # TODO: Predict the action using the greedy policy
            action = network.predict_actions(torch.tensor(state).to(network.device), greedy=True)[0].cpu().numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    venv = gym.make_vec(env.spec.id, args.envs, gym.VectorizeMode.ASYNC)

    # Training
    next_obs, autoreset = venv.reset(seed=args.seed)[0], np.zeros(args.envs, dtype=bool)
    next_done = np.zeros(args.envs, dtype=bool)
    training = True
    iteration = 0
    #step = 0
    #wandb.log({"step": step})

    obs = np.zeros((args.worker_steps, args.envs) + env.observation_space.shape)
    actions = np.zeros((args.worker_steps, args.envs) + env.action_space.shape)
    logprobs = np.zeros((args.worker_steps, args.envs))
    rewards = np.zeros((args.worker_steps, args.envs))
    dones = np.zeros((args.worker_steps, args.envs))
    values = np.zeros((args.worker_steps, args.envs))

    while training:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[args.worker_steps, args.envs]`.
        for step in range(args.worker_steps):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _ = network.predict_actions(torch.tensor(next_obs).to(network.device))
                values[step] = network.predict_values(torch.tensor(next_obs).to(network.device))[0].cpu().numpy()

            action = action.cpu().numpy()
            logprob = logprob.cpu().numpy()
            actions[step] = action
            logprobs[step] = logprob

            # Perform the step
            next_obs, reward, terminated, truncated, _ = venv.step(action)
            next_done = terminated | truncated
            rewards[step] = reward

            #wandb.log({"step": step})

        wandb.log({"average_value": np.mean(values)})
        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to process episodes of individual workers independently, and note that
        # each worker might have generated multiple episodes, the last one probably unfinished.
        with torch.no_grad():
            next_value = network.predict_values(torch.tensor(next_obs).to(network.device))[0].cpu().numpy()
            advantages = np.zeros((args.worker_steps, args.envs))
            lastgaelam = 0
            for t in reversed(range(args.worker_steps)):
                if t == args.worker_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.trace_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        # TODO: Train for `args.epochs` using the collected data. In every epoch,
        # you should randomly sample batches of size `args.batch_size` from the collected data.
        # A possible approach is to create a dataset of `(states, actions, action_probs, advantages, returns)`
        # quintuples using a single `torch.utils.data.StackDataset` and then use a dataloader.

        # Join first two dims

        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        total_size = args.envs * args.worker_steps
        b_inds = np.arange(total_size)
        for epoch in range(args.epochs):
            np.random.shuffle(b_inds)
            for start in range(0, total_size, args.batch_size):
                end = start + args.batch_size
                inds = b_inds[start:end]

                network.train(
                    torch.tensor(b_obs[inds], dtype=torch.float32).to(network.device),
                    torch.tensor(b_actions[inds], dtype=torch.float32).to(network.device),
                    torch.tensor(b_logprobs[inds], dtype=torch.float32).to(network.device),
                    torch.tensor(b_advantages[inds], dtype=torch.float32).to(network.device),
                    torch.tensor(b_returns[inds], dtype=torch.float32).to(network.device)
                )

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            wandb.log({"return": np.mean(returns)})
            print(f"Iteration {iteration}, average return {np.mean(returns)}")

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    wandb.init(project=args.env)
    wandb.config.update(args)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)