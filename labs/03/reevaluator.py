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
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=57, type=int, help="Random seed.")
parser.add_argument("--models_path", default="data/models/lunar_lander", type=str, help="Path to save best models to")


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

    def save_model(Q: np.ndarray, eval_return: float):
        with open(Path(args.models_path) / f"{eval_return:.5f}.pkl", "wb") as f:
            pickle.dump(Q, f)
        with open(Path(args.models_path) / f"{eval_return:.5f}.json", "w") as f:
            json.dump(vars(args), f)
        print(f"Saved model with evaluation return {eval_return:.5f}")

    def remove_model(eval_return: float):
        os.remove(Path(args.models_path) / f"{eval_return:.5f}.pkl")
        try:
            os.remove(Path(args.models_path) / f"{eval_return:.5f}.json")
        except FileNotFoundError:
            pass
        print(f"Removed model with evaluation return {eval_return:.5f}")

    def rename_model(old_return: float, new_return: float):
        os.rename(Path(args.models_path) / f"{old_return:.5f}.pkl", Path(args.models_path) / f"{new_return:.5f}.pkl")
        try:
            os.rename(Path(args.models_path) / f"{old_return:.5f}.json", Path(args.models_path) / f"{new_return:.5f}.json")
        except FileNotFoundError:
            pass
        print(f"Renamed model with evaluation return {old_return:.5f} to {new_return:.5f}")

    def reevaluate_all_models():
        models = get_sorted_models()
        for model in models:
            with open(model, "rb") as f:
                Q = pickle.load(f)
            score = evaluate(Q, 100)
            if score<90:
                remove_model(float(model.stem))
                continue
            print(f"Model {model.stem} scored {score:.2f}")
            slow_score = evaluate(Q, 1000)
            rename_model(float(model.stem), slow_score)



    reevaluate_all_models()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, args.render_each)

    main(env, args)
