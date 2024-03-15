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
from collections import deque

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=57, type=int, help="Random seed.")
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--alpha_final", default=0.001, type=float, help="Final learning rate.")
parser.add_argument("--alpha_final_at", default=10000, type=int, help="Training episodes.")
parser.add_argument("--episodes", default=1000000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration epsilon factor.")
parser.add_argument("--epsilon_final_at", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="tree_backup", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of steps in an episode otherwise punished")

parser.add_argument("--models_path", default="data/models/lunar_lander", type=str, help="Path to save best models to")
parser.add_argument("--best_model_path", default="best_model.pkl", type=str, help="Path to the best model")


def epsilon_greedy(epsilon: float, values: np.ndarray):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(values)-1)
    return values.argmax(-1)

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

    generator = np.random.RandomState()

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

    def save_model(Q: np.ndarray, eval_return: float):
        with open(Path(args.models_path) / f"{eval_return:.5f}.pkl", "wb") as f:
            pickle.dump(Q, f)
        with open(Path(args.models_path) / f"{eval_return:.5f}.json", "w") as f:
            json.dump(vars(args), f)
        print(f"Saved model with evaluation return {eval_return:.5f}")

    def consider_best(Q: np.ndarray):
        models = get_sorted_models()
        fast_eval = evaluate(Q, 100)

        if len(models) < 5 or float(models[-5].stem) < fast_eval:
            slow_eval = evaluate(Q, 400)
            save_model(Q, slow_eval)

    start_time = time.time()

    alpha = args.alpha
    epsilon = args.epsilon

    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))
    #Q1 = load_kth_best_model(1)
    #Q2 = load_kth_best_model(1)

    for _ in range(args.episodes):
        if env.episode % 500 == 0 and env.episode > 0:
            consider_best((Q1+Q2)/2)
            print(f"Episode {env.episode}/{args.episodes}, epsilon {epsilon:.3f}, alpha {alpha:.3f}, elapsed {time.time() - start_time:.1f}s")

        next_state, done = env.reset()[0], False

        next_action = epsilon_greedy(epsilon, Q1[next_state, :] + Q2[next_state, :])

        t = 0
        T = 9999999999999999 # finished time step
        tau = -1
        states, actions, rewards = [deque(maxlen=args.n) for _ in range(3)]
        while t-args.n < T-1:
            if t < T:
                action, state = next_action, next_state
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if not done:
                    action = epsilon_greedy(epsilon, Q1[state, :] + Q2[state, :])
                else:
                    T = t + 1

                if t>args.max_steps:
                    reward -= 10

                states.appendleft(state)
                actions.appendleft(action)
                rewards.appendleft(reward)
            else:
                states.pop()
                actions.pop()
                rewards.pop()

            selected_id = np.random.randint(2)

            if t>=args.n-1:
                #target_policy = compute_target_policy(Q2 if selected_id == 0 else Q1)
                if t+1 >= T:
                    G = reward
                else:
                    best_action = argmax_with_tolerance(Q2[next_state] if selected_id == 0 else Q1[next_state])
                    G = reward + args.gamma * (Q1[next_state][best_action] if selected_id == 0 else Q2[next_state][best_action])

                for k in range(len(states)-1):
                    G = rewards[k+1] + args.gamma * target_policy[states[k]][actions[k]] * G
                    for action in range(env.action_space.n):
                        if action != actions[k]:
                            G += args.gamma * target_policy[states[k]][action] * (Q1[states[k]][action] if selected_id == 0 else Q2[states[k]][action])

                if selected_id == 0:
                    Q2[states[-1]][actions[-1]] += alpha * (G - Q2[states[-1]][actions[-1]])
                else:
                    Q1[states[-1]][actions[-1]] += alpha * (G - Q1[states[-1]][actions[-1]])

            t += 1

        epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
        alpha = np.interp(env.episode + 1, [0, args.alpha_final_at], [args.alpha, args.alpha_final])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, args.render_each)

    main(env, args)
