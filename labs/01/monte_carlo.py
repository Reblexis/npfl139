#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=3000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    C = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(args.episodes):
        # TODO: Perform an episode, collecting states, actions and rewards.

        states, actions, rewards = [], [], []

        state, done = env.reset()[0], False

        while not done:
            # TODO: Compute `action` using epsilon-greedy policy.
            action = np.argmax(Q[state]) if np.random.rand() > args.epsilon else env.action_space.sample()

            # Perform the action.
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            done = terminated or truncated

            state = next_state

        # TODO: Compute returns from the received rewards and update Q and C.
        G = 0
        for i in reversed(range(len(states))):
            G = rewards[i] + G
            C[states[i], actions[i]] += 1
            Q[states[i], actions[i]] += (G - Q[states[i], actions[i]]) / C[states[i], actions[i]]

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed, args.render_each)

    main(env, args)
