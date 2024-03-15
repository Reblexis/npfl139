#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import random
from collections import deque
import pickle

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--episodes", default=500000, type=float, help="Training episodes.")
parser.add_argument("--state_limit", default=10000, type=int, help="Maxikum amount of allowed states to finish, if not "
                                                                   "a big negative reward is received.")
parser.add_argument("--n_step", default=20, type=int, help="Number of steps for n-step updates")
parser.add_argument("--evaluation_episodes", default=1000, type=int, help="Number of evaluation episodes.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="How often evaluate.")
parser.add_argument("--evaluate_after", default=450000, type=int, help="After which episode start evaluating.")
parser.add_argument("--load_model", default=False, type=bool, help="Whether load saved weights.")
parser.add_argument("--start_from", default=160000, type=int, help="When to start training.")


def epsilon_greedy(epsilon: float, values: np.ndarray):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(values)-1)
    return values.argmax(-1)


def evaluate(env: wrappers.EvaluationEnv, args: argparse.Namespace, q) -> float:
    returns = []
    print("Evaluation")
    for i in range(args.evaluation_episodes):
        state, done = env.reset()[0], False
        g = 0
        while not done:
            action = epsilon_greedy(0, q[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            g += reward

            state = next_state
        returns.append(g)
    print(f"finished evaluation, score: {sum(returns) / len(returns)}")
    return sum(returns) / len(returns)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    random_identifier = random.randint(0,1000000)
    # Set random seed
    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        with open('best_q_506632', 'rb') as f:
            q = pickle.load(f)
        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                action = epsilon_greedy(0, q[state, :])
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    else:
        q1 = np.zeros((env.observation_space.n, env.action_space.n))
        q2 = np.zeros((env.observation_space.n, env.action_space.n))
        if args.load_model:
            with open('best_q_506632', 'rb') as f:
                q = pickle.load(f)
                q1 = q/2
                q2 = q/2
            evaluate(env, args, q)

        cur_epsilon = args.epsilon
        cur_alpha = args.alpha
        rewards = deque(maxlen=args.n_step)
        actions = deque(maxlen=args.n_step)
        states = deque(maxlen=args.n_step)
        next_states = deque(maxlen=args.n_step)

        max_score = 0
        for i in range(args.start_from, args.episodes):
            if i > args.evaluate_after and i % args.evaluate_each == 0:
                print(f"Evaluating step: {i}.")
                score = evaluate(env, args, q1+q2)
                if score > max_score:
                    with open("best_q_"+str(random_identifier), "wb") as fp:
                        pickle.dump((q1+q2), fp)
                    print(f"Achieved new best: {score} at {i} step!")
                    max_score = score
            cur_epsilon = args.epsilon - (args.epsilon / args.episodes) * i
            print(cur_epsilon)
            cur_alpha = args.alpha - (args.alpha / args.episodes) * i * 0.98
            state, done = env.reset()[0], False
            num_states = 1
            while not done:
                action = epsilon_greedy(cur_epsilon, q1[state, :] + q2[state, :])
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if num_states > args.state_limit:
                    reward += -10

                rewards.appendleft(reward)
                actions.appendleft(action)
                states.appendleft(state)
                next_states.appendleft(next_state)

                if num_states >= args.n_step:
                    g = 0
                    for j in range(args.n_step):
                        cur_reward = rewards[j]
                        g = cur_reward + g * args.gamma

                    if random.uniform(0, 1) > 0.5:
                        q1[states[-1], actions[-1]] += cur_alpha * (
                                g + args.gamma * q2[next_states[-1], q1[next_states[-1], :].argmax(-1)] -
                                q1[states[-1], actions[-1]])
                    else:
                        q2[states[-1], actions[-1]] += cur_alpha * (
                                g + args.gamma * q1[next_states[-1], q2[next_states[-1], :].argmax(-1)] -
                                q2[states[-1], actions[-1]])

                if random.uniform(0, 1) > 0.5:
                    q1[state, action] += cur_alpha * (args.n_step / 2) * (
                            reward + args.gamma * q2[next_state, q1[next_state, :].argmax(-1)] -
                            q1[state, action])
                else:
                    q2[state, action] += cur_alpha * (args.n_step / 2) * (
                            reward + args.gamma * q1[next_state, q2[next_state, :].argmax(-1)] -
                            q2[state, action])

                state = next_state
                num_states += 1


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, args.render_each)

    main(env, args)