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
parser.add_argument("--alpha", default=1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=5000, type=int, help="Training episodes.")
parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon
    alpha = args.alpha

    training = True
    while training:
        # Perform episode
        if env.episode >= args.epsilon_final_at*2:
            training = False
        state, done = env.reset()[0], False
        while not done:
            # TODO: Choose an action.
            action = np.argmax(np.sum(W[state], axis=0)) if np.random.rand() > epsilon else env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TODO: Update the action-value estimates

            next_action = np.argmax(np.sum(W[next_state], axis=0))
            W[state, action] += alpha * (reward + args.gamma * np.sum(W[next_state, next_action])
                                         - np.sum(W[state, action]))

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
            alpha = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.alpha, args.alpha_final])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose (greedy) action
            action = np.argmax(np.sum(W[state], axis=0))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0"), tiles=args.tiles),
                                 args.seed, args.render_each)

    main(env, args)
