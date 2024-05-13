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
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.15, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=32, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=7, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=512, type=int, help="Steps for each worker to perform.")


class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace) -> None:
        self._args = args

        # TODO: Create an actor using a single hidden layer with `args.hidden_layer_size`
        # units and ReLU activation, produce a policy with `action_space.n` discrete actions.
        self._actor = torch.nn.Sequential(
            torch.nn.Linear(np.prod(observation_space.shape), args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, action_space.n),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)

        # TODO: Create a critic (value predictor) consisting of a single hidden layer with
        # `args.hidden_layer_size` units and ReLU activation, and and output layer with a single output.
        self._critic = torch.nn.Sequential(
            torch.nn.Linear(np.prod(observation_space.shape), args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=args.learning_rate)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args.learning_rate)

    def save_actor(self, path: str):
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str):
        self._actor.load_state_dict(torch.load(path, map_location=self.device))

    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, action_probs: torch.Tensor,
              advantages: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform a single training step of the PPO algorithm.
        # For the policy model, the sum is the sum of:
        # - the PPO loss, where `self._args.clip_epsilon` is used to clip the probability ratio
        # - the entropy regularization with coefficient `self._args.entropy_regularization`.
        #   You can compute it for example using the `torch.distributions.Categorical` class.
        actor_predictions = self._actor(states)
        critic_predictions = self._critic(states)

        ratio = actor_predictions[torch.arange(len(actor_predictions)), actions] / action_probs
        clipped_ratio = torch.clamp(ratio, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon)
        ppo_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        entropy = torch.distributions.Categorical(actor_predictions).entropy().mean()
        entropy_loss = -self._args.entropy_regularization * entropy

        actor_loss = ppo_loss + entropy_loss

        # TODO: The critic model is trained in a stadard way, by using the MSE
        # error between the predicted value function and target returns.

        critic_loss = torch.nn.functional.mse_loss(critic_predictions.squeeze(), returns.squeeze())

        # Perform the optimization step
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        with torch.no_grad():
            return self._actor(states)

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return estimates of value function.
        with torch.no_grad():
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
            action = np.argmax(network.predict_actions(np.array([state])))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    venv = gym.make_vec(env.spec.id, args.envs, gym.VectorizeMode.ASYNC)

    # Training
    state, autoreset = venv.reset(seed=args.seed)[0], np.zeros(args.envs, dtype=bool)
    training = True
    iteration = 0
    while training:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[args.worker_steps, args.envs]`.
        states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
        values.append(network.predict_values(state))
        for _ in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.envs` actions, each
            # sampled from the corresponding policy generated by the `network.predict`
            # executed on the vector `state`.
            probs = network.predict_actions(state)
            action = np.array([np.random.choice(env.action_space.n, p=probs[i]) for i in range(len(probs))])

            # Perform the step
            next_state, reward, terminated, truncated, _ = venv.step(action)
            done = terminated | truncated

            # TODO: Compute and collect the required quantities
            states.append(state)
            actions.append(action)
            action_probs.append(probs[range(len(probs)), action])
            rewards.append(reward)
            dones.append(done)
            values.append(network.predict_values(next_state))

            autoreset = done
            state = next_state

        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to process episodes of individual workers independently, and note that
        # each worker might have generated multiple episodes, the last one probably unfinished.
        advantages, returns = [], []
        for i in range(args.envs):
            advantage, ret = [], []
            g = 0
            for t in range(len(states) - 1, -1, -1):
                td_error = rewards[t][i] + (1 - dones[t][i]) * args.gamma * values[t + 1][i] - values[t][i]
                g = td_error + (1-dones[t][i]) * args.gamma * args.trace_lambda * g
                advantage.append(g)
                ret.append(g + values[t][i])
            advantages.append(advantage[::-1])
            returns.append(ret[::-1])

        # TODO: Train for `args.epochs` using the collected data. In every epoch,
        # you should randomly sample batches of size `args.batch_size` from the collected data.
        # A possible approach is to create a dataset of `(states, actions, action_probs, advantages, returns)`
        # quintuples using a single `torch.utils.data.StackDataset` and then use a dataloader.
        advantages_np = np.array(advantages)
        returns_np = np.array(returns)
        returns_transposed = np.transpose(returns_np, (1, 0, 2))
        advantages_transposed = np.transpose(advantages_np, (1, 0, 2))
        states_concat = np.concatenate(states)
        actions_concat = np.concatenate(actions)
        action_probs_concat = np.concatenate(action_probs)
        advantages_concat = np.concatenate(advantages_transposed)
        returns_concat = np.concatenate(returns_transposed)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(states_concat, dtype=torch.float32),
            torch.tensor(actions_concat, dtype=torch.int64),
            torch.tensor(action_probs_concat, dtype=torch.float32),
            torch.tensor(advantages_concat, dtype=torch.float32),
            torch.tensor(returns_concat, dtype=torch.float32),
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for _ in range(args.epochs):
            for batch in dataloader:
                states, actions, action_probs, advantages, returns = batch
                network.train(states, actions, action_probs, advantages, returns)

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            print(f"Iteration {iteration}, average return {np.mean(returns)}")

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
