#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=8, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")


class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self.policy_model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
        ).to(self.device)

        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=args.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=args.learning_rate)

        self.policy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.value_loss = torch.nn.MSELoss()

        self.policy_model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.value_model.apply(wrappers.torch_init_with_xavier_and_zeros)

    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Train the policy network using policy gradient theorem
        # and the value network using MSE.
        #
        # The `args.entropy_regularization` might be used to include actor
        # entropy regularization -- the assignment can be solved even without
        # it, but my reference solution learns more quickly when using it.
        # In any case, `torch.distributions.Categorical` is the suitable distribution
        # offering the `.entropy()` method.
        self.policy_model.train()
        self.policy_optimizer.zero_grad()

        self.value_model.train()
        self.value_optimizer.zero_grad()

        values = self.value_model(states).squeeze()
        values_no_grad = values.detach()
        value_loss = self.value_loss(values, returns)
        value_loss.backward()

        logits = self.policy_model(states)
        policy_loss = self.policy_loss(logits, actions) @ (returns - values_no_grad)
        policy_loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        self.policy_model.eval()
        with torch.no_grad():
            logits = self.policy_model(states)
            policy = torch.nn.functional.softmax(logits, dim=-1)
            return policy

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        self.value_model.eval()
        with torch.no_grad():
            values = self.value_model(states).squeeze()
            return values


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            action = np.argmax(network.predict_actions(state))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC)
    states = vector_env.reset(seed=args.seed)[0]

    training, autoreset = True, np.zeros(args.envs, dtype=bool)
    while training:
        # Training
        for i in range(args.evaluate_each):
            # TODO: Choose actions using `network.predict_actions`.
            actions = np.array([np.random.choice(env.action_space.n, p=network.predict_actions(state)) for state in states])

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            # TODO: Compute estimates of returns by one-step bootstrapping

            estimated_returns = rewards + args.gamma * network.predict_values(next_states) * ~dones

            # TODO: Train network using current states, chosen actions and estimated returns.
            # However, note that when `autoreset[i] == True`, the `i`-th environment has
            # just reset, so `states[i]` is the terminal state of a previous episode
            # and `nextstate` is the initial state of a new episode.

            network.train(states, actions, estimated_returns)

            states = next_states
            autoreset = dones

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        if np.mean(returns) > 490:
            break

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
