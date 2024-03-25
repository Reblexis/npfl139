#!/usr/bin/env python3
import argparse
import collections
from typing import Self, Any

import cv2
import wandb

from numpy import signedinteger

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
# Meta
parser.add_argument("--n_steps", default=100000, type=int, help="Number of training steps.")
parser.add_argument("--evaluate_for", default=15, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--evaluate_each", default=5, type=int, help="Evaluate each number of network updates.")

# Preprocessor parameters
parser.add_argument("--image_size", default=84, type=int, help="Size of the preprocessed image.")

# DQN parameters
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=2e-4, type=float, help="Learning rate.")
parser.add_argument("--replay_buffer_max_length", default=1000000, type=int, help="Maximum replay buffer length.")
parser.add_argument("--replay_buffer_min_length", default=10000, type=int, help="Minimal replay buffer size.")
parser.add_argument("--target_update_frequency", default=10000, type=int,
                    help="Frequency of target network update (in steps).")

parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=50000, type=int,
                    help="Number of steps until the final exploration factor.")

# Agent parameters
parser.add_argument("--episode_chunks_size", default=1, type=int,
                    help="Number of episodes to simulate before training.")

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,env.action_space.n),
        )

    def forward(self, x):
        return self._model(x)


class Network:
    # Use GPU if available.
    def __init__(self, args: argparse.Namespace) -> None:
        self._model = Model().to(DEVICE)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

        self._loss = torch.nn.SmoothL1Loss()

        self._model.apply(wrappers.torch_init_with_xavier_and_zeros)

    def train(self, states: np.ndarray, q_values: np.ndarray) -> float:
        states = torch.tensor(states, device=DEVICE, dtype=torch.float32)
        q_values = torch.tensor(q_values, device=DEVICE, dtype=torch.float32)
        self._model.train()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

        return loss.item()

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = torch.tensor(states, device=DEVICE, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            return np.array(self._model(states).cpu())

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: Self) -> None:
        self._model.load_state_dict(other._model.state_dict())


class Preprocessor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return state


class Episode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state: np.ndarray, action: signedinteger[Any], reward: float, next_state: np.ndarray,
            done: bool) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> tuple[np.ndarray, signedinteger[Any], float, np.ndarray, bool]:
        return self.states[index], self.actions[index], self.rewards[index], self.next_states[index], self.dones[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DQN:

    def __init__(self, network: Network, _args: argparse.Namespace) -> None:
        self.n_steps = _args.n_steps

        self.policy_network = network
        self.target_network = Network(_args)
        self.target_network.copy_weights_from(self.policy_network)

        self.replay_buffer = wrappers.ReplayBuffer(max_length=_args.replay_buffer_max_length)

        self.min_replay_buffer_size = _args.replay_buffer_min_length
        self.batch_size = _args.batch_size

        self.target_update_frequency = _args.target_update_frequency

        self.gamma = _args.gamma
        self.orig_epsilon = _args.epsilon
        self.epsilon = _args.epsilon
        wandb.log({"epsilon": self.epsilon})
        self.epsilon_final = _args.epsilon_final
        self.epsilon_final_at = _args.epsilon_final_at

        self.current_episode = 0
        self.current_step = 0
        wandb.log({"step": self.current_step})
        wandb.log({"episode": self.current_episode})
        self.last_updated_episode = 0
        self.last_updated_step = 0

        self.network_updates = 0
        wandb.log({"network_updates": self.network_updates})

        self.target_network_updates = 0
        wandb.log({"target_network_updates": self.target_network_updates})

    def get_action(self, state: np.ndarray, greedy: bool = False) -> signedinteger[Any]:
        return np.random.randint(2) if (np.random.rand() < self.epsilon and not greedy) else np.argmax(
            self.policy_network.predict(state[np.newaxis])[0])

    def collect_episodes(self, episodes: list[Episode]) -> None:
        for episode in episodes:
            for state, action, reward, next_state, done in episode:
                self.replay_buffer.append(Transition(state, action, reward, done, next_state))
                self.current_step += 1
                if self.current_step >= self.n_steps:
                    return
                wandb.log({"step": self.current_step})

            self.current_episode += 1
            wandb.log({"episode": self.current_episode})

    def update(self) -> None:
        self.epsilon = np.interp(self.current_step, [0, self.epsilon_final_at], [self.orig_epsilon, self.epsilon_final])
        wandb.log({"epsilon": self.epsilon})

        step_diff = self.current_step - self.last_updated_step
        episode_diff = self.current_episode - self.last_updated_episode

        self.last_updated_step = self.current_step
        self.last_updated_episode = self.current_episode

        if len(self.replay_buffer) < self.min_replay_buffer_size:
            return

        losses = []
        q_values_means = []

        for _ in range(step_diff):
            if self.network_updates % self.target_update_frequency == 0:
                self.target_network_updates += 1
                wandb.log({"target_network_updates": self.target_network_updates})
                self.target_network.copy_weights_from(self.policy_network)

            transitions = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, dones, next_states = zip(*transitions)

            states = np.array(states)
            next_states = np.array(next_states)

            q_values = self.policy_network.predict(states)
            network_next_q_values = self.policy_network.predict(next_states)
            target_next_q_values = self.target_network.predict(next_states)

            for i, (_state, _action, _reward, _done, _next_state) in enumerate(transitions):
                q_values[i][_action] = (_reward + self.gamma * (1 - _done) *
                                        target_next_q_values[i][np.argmax(network_next_q_values[i])])

            loss = self.policy_network.train(states, q_values)
            losses.append(loss)
            q_values_means.append(np.mean(q_values))

            self.network_updates += 1
            wandb.log({"network_updates": self.network_updates})

        wandb.log({"loss": np.mean(losses)})
        wandb.log({"q_values_mean": np.mean(q_values_means)})

    def learn(self, episodes: list[Episode]) -> None:
        self.collect_episodes(episodes)
        if self.current_step < self.n_steps:
            self.update()


class Agent:
    def __init__(self, _env: wrappers.EvaluationEnv, preprocessor: Preprocessor, strategy: DQN,
                 _args: argparse.Namespace) -> None:
        self.env = _env
        self.preprocessor = preprocessor
        self.strategy = strategy

        self.evaluate_each = _args.evaluate_each
        self.evaluate_for = _args.evaluate_for

        self.episode_chunks_size = _args.episode_chunks_size

    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        return self.preprocessor(state)

    def simulate(self, num_episodes: int, greedy: bool = False) -> list[Episode]:
        episodes = []

        for k in range(num_episodes):
            state, done = self.env.reset()[0], False
            state = self.preprocess_state(state)
            episode = Episode()
            reward_sum = 0
            while not done:
                action = self.strategy.get_action(state, greedy)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                reward_sum += reward
                next_state = self.preprocess_state(next_state)
                done = terminated or truncated

                episode.add(state, action, float(reward), next_state, done)

                state = next_state

            episodes.append(episode)
        return episodes

    def evaluate(self) -> float:
        episodes = self.simulate(self.evaluate_for, greedy=True)
        rewards = [sum(episode.rewards) for episode in episodes]
        return sum(rewards) / len(rewards)

    def train(self) -> None:
        i = 0
        while self.strategy.current_step < self.strategy.n_steps:
            episodes = self.simulate(self.episode_chunks_size)
            self.strategy.learn(episodes)
            if self.strategy.current_step > self.strategy.n_steps:
                break

            if i % self.evaluate_each == 0:
                print(f"Train step {i}")
                wandb.log({"eval_reward": self.evaluate()})
            i += 1


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Create the network
    agent = Agent(env, Preprocessor(args), DQN(Network(args), args), args)
    agent.train()

    # Final evaluation
    # Save network
    # network._model.load_state_dict(torch.load("model.pth", map_location=network.device))

    #torch.save(network._model.state_dict(), "model.pth")

    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose (greedy) action
            action = agent.strategy.get_action(agent.preprocess_state(state), greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    wandb.init(
        project="cartpole",
        config=vars(args),
    )

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
