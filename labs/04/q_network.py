#!/usr/bin/env python3
import argparse
import collections
from typing import Self

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
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.9, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=500000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=10000, type=int, help="Target update frequency.")

parser.add_argument("--replay_buffer_max_length", default=1000000, type=int, help="Maximum replay buffer length.")
parser.add_argument("--replay_buffer_min_length", default=50000, type=int, help="Minimal replay buffer length.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluation_episodes", default=100, type=int, help="Evaluate each number of episodes.")


class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`.
        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], env.action_space.n),
        ).to(self.device)

        # TODO: Define an optimizer (most likely from `torch.optim`).
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=args.learning_rate)

        # TODO: Define the loss (most likely some `torch.nn.*Loss`).
        self._loss = torch.nn.MSELoss()

        # PyTorch uses uniform initializer $U[-1/sqrt n, 1/sqrt n]$ for both weights and biases.
        # Keras uses Glorot (also known as Xavier) uniform for weights and zeros for biases.
        # In some experiments, the Keras initialization works slightly better for RL,
        # so we use it instead of the PyTorch initialization; but feel free to experiment.
        self._model.apply(wrappers.torch_init_with_xavier_and_zeros)

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    #
    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @wrappers.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return self._model(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: Self) -> None:
        self._model.load_state_dict(other._model.state_dict())


def evaluate(env: wrappers.EvaluationEnv, network: Network, args: argparse.Namespace) -> float:
    returns = []
    for _ in range(args.evaluation_episodes):
        state, done = env.reset(start_evaluation=False)[0], False
        g = 0
        while not done:
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            g += reward

        returns.append(g)
    return np.sum(returns) / args.evaluation_episodes

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Construct the network
    network = Network(env, args)
    target_network = Network(env, args)

    # Replay memory; the `max_length` parameter can be passed to limit its size.
    replay_buffer = wrappers.ReplayBuffer(max_length=args.replay_buffer_max_length)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    training = True
    steps = 0
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            q_values = network.predict(state[np.newaxis])[0]
            action = np.random.randint(env.action_space.n) if np.random.rand() < epsilon else np.argmax(q_values)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the `replay_buffer` is large enough, perform training using
            # a batch of `args.batch_size` uniformly randomly chosen transitions.
            #
            # The `replay_buffer` offers a method with signature
            #   sample(self, size, generator=np.random, replace=True) -> list[Transition]
            # which returns uniformly selected batch of `size` transitions, either with
            # replacement (which is much faster, and hence the default) or without.
            # By default, `np.random` is used to generate the random indices, but you can
            # pass your own `np.random.RandomState` instance.

            # After you compute suitable targets, you can train the network by
            #   network.train(...)

            if len(replay_buffer) >= args.replay_buffer_min_length:
                transitions = replay_buffer.sample(args.batch_size)
                states, actions, rewards, dones, next_states = zip(*transitions)

                q_values = network.predict(states)
                network_next_q_values = network.predict(next_states)
                target_next_q_values = target_network.predict(next_states)

                for i, (_state, _action, _reward, _done, _next_state) in enumerate(transitions):
                    q_values[i][_action] = (_reward + args.gamma * (1 - _done) *
                                            target_next_q_values[i][np.argmax(network_next_q_values[i])])

                network.train(states, q_values)

            if steps % args.target_update_freq == 0:
                print(f"Currect step: {steps}, updating target network.")
                target_network.copy_weights_from(network)

            state = next_state
            steps += 1

            if args.epsilon_final_at:
                epsilon = np.interp(steps, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        if steps>args.epsilon_final_at and steps % args.evaluate_each == 0:
            evaluation_score = evaluate(env, network, args)
            print(f"Current step: {steps}, evaluation score: {evaluation_score}")
            if evaluation_score > 460:
                training = False

    # Final evaluation
    # Save network
    #network._model.load_state_dict(torch.load("model.pth", map_location=network.device))
    torch.save(network._model.state_dict(), "model.pth")

    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose (greedy) action
            action = np.argmax(network.predict(state[np.newaxis])[0])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
