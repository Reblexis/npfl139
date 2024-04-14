#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=8, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=16, type=int, help="Tiles to use.")


class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Analogously to paac, your model should contain two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # The given states are tile encoded, so they are integer indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with `args.tiles` ones).
        # (Or you could even use `torch.nn.EmbeddingBag`, but be careful
        # about the range of the initial weights.)
        #
        # The actor computes `mus` and `sds`, each of shape `[batch_size, actions]`.
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and a ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `torch.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `torch.exp` or `torch.nn.functional.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.

        self.policy_mus_model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
            torch.nn.Tanh(),
        ).to(self.device)

        self.policy_sds_model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
            torch.nn.Softplus(),
        ).to(self.device)

        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)

        self.policy_mus_optimizer = torch.optim.Adam(self.policy_mus_model.parameters(), lr=args.learning_rate)
        self.policy_sds_optimizer = torch.optim.Adam(self.policy_sds_model.parameters(), lr=args.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=args.learning_rate)

        self.value_loss = torch.nn.MSELoss()

        self.policy_mus_model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.policy_sds_model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.value_model.apply(wrappers.torch_init_with_xavier_and_zeros)

    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.int64, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Run the model on given `states` and compute
        # `sds`, `mus` and predicted values. Then create `action_distribution` using
        # `torch.distributions.Normal` class and the computed `mus` and `sds`.
        #
        # TODO: Train the actor using the sum of the following two losses:
        # - REINFORCE loss, i.e., the negative log likelihood of the `actions` in the
        #   `action_distribution` (using the `log_prob` method). You then need to sum
        #   the log probabilities of the action components in a single batch example.
        #   Finally, multiply the resulting vector by `(returns - predicted values)`
        #   and compute its mean. Note that the gradient must not flow through
        #   the predicted values (you can use `.detach()` if necessary).
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.

        self.policy_mus_model.train()
        self.policy_mus_optimizer.zero_grad()

        self.policy_sds_model.train()
        self.policy_sds_optimizer.zero_grad()

        self.value_model.train()
        self.value_optimizer.zero_grad()

        mus = self.policy_mus_model(states)
        sds = self.policy_sds_model(states)
        values = self.value_model(states).squeeze()

        action_distribution = torch.distributions.Normal(mus, sds)
        action_log_probs = action_distribution.log_prob(actions).sum(dim=1)
        advantage = returns - values

        policy_loss = -(action_log_probs * advantage).mean() - args.entropy_regularization * action_distribution.entropy().mean()
        policy_loss.backward()

        self.policy_mus_optimizer.step()
        self.policy_sds_optimizer.step()

        value_loss = self.value_loss(values, returns)
        value_loss.backward()

    @wrappers.typed_torch_function(device, torch.int64)
    def predict_actions(self, states: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.policy_mus_model.eval()
        self.policy_sds_model.eval()
        with torch.no_grad():
            mus = self.policy_mus_model(states)
            sds = self.policy_sds_model(states)
            return mus, sds

    @wrappers.typed_torch_function(device, torch.int64)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        self.value_model.eval()
        with torch.no_grad():
            values = self.value_model(states).squeeze()
            return values

def preprocess_states(states: np.ndarray) -> np.ndarray:
    print(states)
    return states

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
            # TODO: Predict the action using the greedy policy.
            action_info = network.predict_actions(state)
            action = np.random.normal(action_info[0], action_info[1])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.make_vec("MountainCarContinuous-v0", args.envs, gym.VectorizeMode.ASYNC,
                              wrappers=[lambda env: wrappers.DiscreteMountainCarWrapper(env, tiles=args.tiles)])
    states = vector_env.reset(seed=args.seed)[0]
    states = preprocess_states(states)

    training, autoreset = True, np.zeros(args.envs, dtype=bool)
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `network.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.
            action_infos = network.predict_actions(states)
            actions = np.array([np.random.normal(mu, sd) for mu, sd in zip(*action_infos)])

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            next_states = preprocess_states(next_states)
            dones = terminated | truncated

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            estimated_returns = rewards + args.gamma * network.predict_values(next_states) * ~dones

            # TODO(paac): Train network using current states, chosen actions and estimated returns.
            # However, note that when `autoreset[i] == True`, the `i`-th environment has
            # just reset, so `states[i]` is the terminal state of a previous episode
            # and `nextstate` is the initial state of a new episode.

            network.train(states, actions, estimated_returns)

            states = next_states
            autoreset = dones

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        if np.mean(returns) > 90:
            break

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles),
        args.seed, args.render_each)

    main(env, args)
