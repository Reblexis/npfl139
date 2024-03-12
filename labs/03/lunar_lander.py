#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import time
import os
from pathlib import Path
import pickle
import json
import random
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=57, type=int, help="Random seed.")
parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate alpha.")
parser.add_argument("--alpha_final", default=0.001, type=float, help="Final learning rate.")
parser.add_argument("--alpha_final_at", default=1000, type=int, help="Training episodes.")
parser.add_argument("--episodes", default=1000000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.05, type=float, help="Exploration epsilon factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration epsilon factor.")
parser.add_argument("--epsilon_final_at", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="tree_backup", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=8, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")

parser.add_argument("--models_path", default="data/models/lunar_lander", type=str, help="Path to save best models to")
parser.add_argument("--best_model_path", default="best_model.pkl", type=str, help="Path to the best model")


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def get_sorted_models() -> list[Path]:
    models_raw = (Path(args.models_path).glob("*.pkl"))
    models = []
    for model in models_raw:
        try:
            float(model.stem)
            models.append(model)
        except ValueError:
            pass

    models = sorted(models, key=lambda x: float(x.stem))

    return models

def load_kth_best_model(k: int) -> np.ndarray:
    models = get_sorted_models()

    if len(models) < k:
        raise ValueError(f"Model {k} does not exist, only {len(models)} models found")

    with open(models[-k], "rb") as f:
        return pickle.load(f)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    #np.random.seed(args.seed)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        with open(Path(args.best_model_path), "rb") as f:
            Q = pickle.load(f)

        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                action = argmax_with_tolerance(Q[state])
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    generator = np.random.RandomState()

    def choose_next_action(Q: np.ndarray) -> tuple[int, float]:
        greedy_action = argmax_with_tolerance(Q[next_state])
        next_action = greedy_action if generator.uniform() >= epsilon else env.action_space.sample()
        return next_action, epsilon / env.action_space.n + (1 - epsilon) * (greedy_action == next_action)

    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - epsilon) * target_policy + epsilon / env.action_space.n
        return target_policy

    def evaluate(Q: np.ndarray, num_episodes) -> float:
        returns = []
        for _ in range(num_episodes):
            state, done = env.reset(start_evaluation=False)[0], False
            G = 0
            while not done:
                action = argmax_with_tolerance(Q[state])
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                G += reward
            returns.append(G)
        return sum(returns) / num_episodes

    def consider_best(Q: np.ndarray):
        models = get_sorted_models()
        fast_eval = evaluate(Q, 10)

        if len(models) < 5 or float(models[-5].stem) < fast_eval:
            slow_eval = evaluate(Q, 20)
            with open(Path(args.models_path) / f"{slow_eval:.5f}.pkl", "wb") as f:
                pickle.dump(Q, f)
            with open(Path(args.models_path) / "hyperparameters.json", "w") as f:
                json.dump(vars(args), f)
            print(f"Saved model with evaluation return {slow_eval:.5f}")

    start_time = time.time()

    alpha = args.alpha
    epsilon = args.epsilon

    Q = load_kth_best_model(1)

    for _ in range(args.episodes):
        if env.episode % 200 == 0 and env.episode > 0:
            consider_best(Q)
            k = random.randint(1, 3)
            Q = load_kth_best_model(k)
            print(f"Episode {env.episode}/{args.episodes}, epsilon {epsilon:.3f}, alpha {alpha:.3f}, elapsed {time.time() - start_time:.1f}s")

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

                Q[states[tau]][actions[tau]] += alpha * (G - Q[states[tau]][actions[tau]]) * w

            t += 1

        epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
        alpha = np.interp(env.episode + 1, [0, args.alpha_final_at], [args.alpha, args.alpha_final])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, args.render_each)

    main(env, args)
