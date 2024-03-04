#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=152, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
parser.add_argument("--episodes", default=15000, type=int, help="Training episodes.")

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # TODO: Variable creation and initialization
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    eps = args.epsilon
    alpha = args.alpha
    cur_episode = 0

    training = True
    while training:
        eps = max(eps * 0.999, args.epsilon_final)
        alpha = max(alpha * 0.999, 0.01)
        cur_episode += 1

        if cur_episode >= args.episodes:
            training = False

        if cur_episode % 100 == 0:
            print(f"Episode {cur_episode}/{args.episodes}, epsilon {eps:.3f}")

        # Perform episode
        state, done = env.reset()[0], False
        returns = 0
        while not done:
            # TODO: Perform an action.
            action = np.argmax(Q[state]) if np.random.rand() > eps else env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TODO: Update the action-value estimates
            Q[state, action] += args.alpha * (reward + args.gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state

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
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), args.seed, args.render_each)

    main(env, args)
