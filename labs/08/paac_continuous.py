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
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=8, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=8, type=int, help="Tiles to use.")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseModule(torch.nn.Module):
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace):
        super().__init__()

        self.linear1 = torch.nn.Linear(env.observation_space.nvec[-1], env.action_space.shape[0])

    def forward(self, x):
        x = self.linear1(x)
        return x


class MeanModule(torch.nn.Module):
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace):
        super().__init__()

        self.actions_min = torch.tensor(env.action_space.low).to(DEVICE)
        self.actions_max = torch.tensor(env.action_space.high).to(DEVICE)

        self.base_module = BaseModule(env, args)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.base_module(x)
        x = self.tanh(x)
        return x


class SdModule(torch.nn.Module):
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace):
        super().__init__()

        self.base_module = BaseModule(env, args)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x = self.base_module(x)
        x = self.softplus(x)
        return x


class Network:
    # Use GPU if available.

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

        self.policy_mus_model = MeanModule(env, args).to(DEVICE)

        self.policy_sds_model = SdModule(env, args).to(DEVICE)

        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.nvec[-1], 1),
        ).to(DEVICE)

        self.policy_mus_optimizer = torch.optim.Adam(self.policy_mus_model.parameters(), lr=args.learning_rate)
        self.policy_sds_optimizer = torch.optim.Adam(self.policy_sds_model.parameters(), lr=args.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=args.learning_rate)

        self.value_loss = torch.nn.MSELoss()

        self.policy_mus_model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.policy_sds_model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.value_model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.entropy_regularization = args.entropy_regularization

    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(DEVICE, torch.float32, torch.float32, torch.float32)
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
        self.policy_mus_model.zero_grad()

        self.policy_sds_model.train()
        self.policy_sds_model.zero_grad()

        self.value_model.train()
        self.value_model.zero_grad()

        mus = self.policy_mus_model(states)
        sds = self.policy_sds_model(states)
        values = self.value_model(states).squeeze()
        values_no_grad = values.detach()
        value_loss = self.value_loss(values, returns)
        value_loss.backward()

        action_distribution = torch.distributions.Normal(mus, sds)
        action_log_probs = action_distribution.log_prob(actions).squeeze()
        advantage = returns - values_no_grad

        policy_loss = -(action_log_probs @ advantage) - self.entropy_regularization * action_distribution.entropy().mean()
        policy_loss.backward()

        self.policy_mus_optimizer.step()
        self.policy_sds_optimizer.step()
        self.value_optimizer.step()

    def save(self):
        torch.save(self.policy_mus_model.state_dict(), "policy_mus_model.pth")
        torch.save(self.policy_sds_model.state_dict(), "policy_sds_model.pth")
        torch.save(self.value_model.state_dict(), "value_model.pth")
        # save cpu versions
        torch.save(self.policy_mus_model.to(torch.device("cpu")).state_dict(), "policy_mus_model_cpu.pth")
        torch.save(self.policy_sds_model.to(torch.device("cpu")).state_dict(), "policy_sds_model_cpu.pth")
        torch.save(self.value_model.to(torch.device("cpu")).state_dict(), "value_model_cpu.pth")

        self.policy_mus_model.to(DEVICE)
        self.policy_sds_model.to(DEVICE)
        self.value_model.to(DEVICE)

    def load(self):
        self.policy_mus_model.load_state_dict(torch.load("policy_mus_model.pth"))
        self.policy_sds_model.load_state_dict(torch.load("policy_sds_model.pth"))
        self.value_model.load_state_dict(torch.load("value_model.pth"))

    def load_cpu(self):
        self.policy_mus_model.load_state_dict(torch.load("policy_mus_model_cpu.pth"))
        self.policy_sds_model.load_state_dict(torch.load("policy_sds_model_cpu.pth"))
        self.value_model.load_state_dict(torch.load("value_model_cpu.pth"))

    @wrappers.typed_torch_function(DEVICE, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.policy_mus_model.eval()
        self.policy_sds_model.eval()
        with torch.no_grad():
            mus = self.policy_mus_model(states)
            sds = self.policy_sds_model(states)
            return mus, sds

    @wrappers.typed_torch_function(DEVICE, torch.float32)
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

    def preprocess_states(states: np.ndarray) -> np.ndarray:
        prep_states = np.zeros((len(states), env.observation_space.nvec[-1]), dtype=np.float32)
        for i, state in enumerate(states):
            prep_states[i, state] = 1.0

        return prep_states

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        state = preprocess_states(np.array([state]))[0]
        while not done:
            # TODO: Predict the action using the greedy policy.
            action_info = network.predict_actions(state)
            action = np.random.normal(action_info[0], action_info[1])
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, terminated, truncated, _ = env.step(action)
            state = preprocess_states(np.array([state]))[0]
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.make_vec("MountainCarContinuous-v0", args.envs, gym.VectorizeMode.ASYNC,
                              wrappers=[lambda env: wrappers.DiscreteMountainCarWrapper(env, tiles=args.tiles)])
    states = vector_env.reset(seed=args.seed)[0]
    states = preprocess_states(states)

    training, autoreset = not args.recodex, np.zeros(args.envs, dtype=bool)
    while training:
        # Training
        for i in range(args.evaluate_each):
            action_infos = network.predict_actions(states)
            actions = np.array([np.random.normal(mu, sd) for mu, sd in zip(*action_infos)])
            actions = np.clip(actions, env.action_space.low, env.action_space.high)

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            next_states = preprocess_states(next_states)
            dones = terminated | truncated

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            estimated_returns = rewards + ~dones * args.gamma * network.predict_values(next_states)

            # TODO(paac): Train network using current states, chosen actions and estimated returns.
            # However, note that when `autoreset[i] == True`, the `i`-th environment has
            # just reset, so `states[i]` is the terminal state of a previous episode
            # and `nextstate` is the initial state of a new episode.

            network.train(states, actions, estimated_returns)

            states = next_states
            autoreset = dones

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        if np.mean(returns) > 92:
            network.save()
            break

    network.load_cpu()
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
