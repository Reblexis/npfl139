#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="tree_backup", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=4, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=True, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed.")


# If you add more arguments, ReCodEx will keep them with your default values.


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a fixed seed
    generator = np.random.RandomState(args.seed)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=min(200, args.episodes))

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_next_action(Q: np.ndarray) -> tuple[int, float]:
        greedy_action = argmax_with_tolerance(Q[next_state])
        next_action = greedy_action if generator.uniform() >= args.epsilon else env.action_space.sample()
        return next_action, args.epsilon / env.action_space.n + (1 - args.epsilon) * (greedy_action == next_action)

    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    for _ in range(args.episodes):
        next_state, done = env.reset()[0], False

        next_action, next_action_prob = choose_next_action(Q)

        t = 0
        T = 9999999999999999 # finished time step
        tau = -1
        states, actions, action_probs, rewards = [], [], [], []
        while tau < T-1:
            if t < T:
                action, action_prob, state = next_action, next_action_prob, next_state
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if not done:
                    next_action, next_action_prob = choose_next_action(Q)
                else:
                    T = t + 1

                states.append(state)
                actions.append(action)
                action_probs.append(action_prob)
                rewards.append(reward)

            tau = t - args.n + 1

            if tau >= 0:
                target_policy = compute_target_policy(Q)
                w = 1
                if t+1 >= T:
                    G = reward
                else:
                    if args.mode == "sarsa":
                        G = reward + args.gamma * Q[next_state][next_action]
                        w = target_policy[next_state][next_action] / next_action_prob
                    elif args.mode == "expected_sarsa" or args.mode == "tree_backup":
                        G = reward + args.gamma * np.sum(Q[next_state] * target_policy[next_state])
                    else:
                        raise ValueError(f"Unknown mode {args.mode}")

                for k in range(min(t-1, T-2), tau-1, -1):
                    if args.mode == "tree_backup":
                        G = rewards[k] + args.gamma * target_policy[states[k+1]][actions[k+1]] * G
                        for action in range(env.action_space.n):
                            if action != actions[k+1]:
                                G += args.gamma * target_policy[states[k+1]][action] * Q[states[k+1]][action]
                    elif args.mode=="sarsa":
                        G = rewards[k] + args.gamma * G
                    elif args.mode == "expected_sarsa":
                        G = rewards[k] + args.gamma * G
                    else:
                        raise ValueError(f"Unknown mode {args.mode}")

                    if args.mode != "tree_backup" and args.off_policy:
                        w *= target_policy[states[k+1]][actions[k+1]] / action_probs[k+1]

                Q[states[tau]][actions[tau]] += args.alpha * (G - Q[states[tau]][actions[tau]]) * w

            t += 1

    return Q


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
