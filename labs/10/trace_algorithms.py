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
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy (less exploratory target)")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--trace_lambda", default=None, type=float, help="Trace factor lambda, if any.")
parser.add_argument("--vtrace_clip", default=None, type=float, help="V-Trace clip rho and c, if any.")
# If you add more arguments, ReCodEx will keep them with your default values.


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def create_env(args: argparse.Namespace, report_each: int = 100, **kwargs) \
        -> tuple[wrappers.EvaluationEnv, np.ndarray, np.ndarray, np.ndarray]:
    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=report_each, **kwargs)

    # Extract a deterministic MDP into three NumPy arrays
    # - R[state][action] is the reward
    # - D[state][action] is the True/False value indicating end of episode
    # - N[state][action] is the next state
    R, D, N = [
        np.array([
            [env.unwrapped.P[s][a][0][i] for a in range(env.action_space.n)] for s in range(env.observation_space.n)])
        for i in [2, 3, 1]
    ]

    return env, R, D, N


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a deterministic MDP, where R, D, N are rewards, dones and
    # next_states for a given state and action.
    env, R, D, N = create_env(args)

    # Create a random seed generator
    generator = np.random.RandomState(args.seed)

    V = np.zeros(env.observation_space.n)

    # The target policy is either the behavior policy (if not `args.off_policy`),
    # or an epsilon/3-greedy policy (if `args.off_policy`).
    def compute_target_policy(V: np.ndarray) -> np.ndarray:
        epsilon = args.epsilon / 3 if args.off_policy else args.epsilon
        greedy_policy = np.eye(env.action_space.n)[argmax_with_tolerance(R + (1 - D) * args.gamma * V[N])]
        return (1 - epsilon) * greedy_policy + epsilon / env.action_space.n * np.ones_like(greedy_policy)

    for _ in range(args.episodes):
        state, done = env.reset()[0], False

        t = 0
        T = 9999999999999999
        tau = -1
        states, next_states, actions, action_probs, rewards, dones = [], [], [], [], [], []

        # Generate episode and update V using the given TD method
        while tau < T-1:
            if t < T:
                best_action = argmax_with_tolerance(R[state] + (1 - D[state]) * args.gamma * V[N[state]])
                action = best_action if generator.uniform() >= args.epsilon else env.action_space.sample()
                action_prob = args.epsilon / env.action_space.n + (1 - args.epsilon) * (action == best_action)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if done:
                    T = t + 1

                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                action_probs.append(action_prob)
                rewards.append(reward)
                dones.append(done)

            tau = t - args.n + 1

            # TODO: Perform the update to the state value function `V`, using
            # a TD update with the following parameters:
            # - `args.n`: use `args.n`-step estimated return
            # - if `args.trace_lambda` is not None, use eligibility traces
            #   (the `args.n`-step truncated `args.trace_lambda`-return)
            # - `args.off_policy`:
            #   - if False, the `args.epsilon`-greedy behaviour policy is also the target policy
            #   - if True, the target policy is an (`args.epsilon`/3)-greedy policy; use
            #     off-policy correction using importance sampling with control variates
            #     - if `args.vtrace_clip` is not None, clip the individual importance sample
            #       ratios with it
            #
            # Perform the updates as soon as you can -- whenever you have all the information
            # to update `V[state]`, do it.
            #
            # When performing off-policy estimation, use `action_prob` at the time of
            # taking the `action` as the behaviour policy action probability, and the
            # `compute_target_policy(V)` with the current `V` as the target policy.
            #
            # Do not forget that when `done` is True, bootstrapping on the
            # `next_state` is not used.
            #
            # Also note that when the episode ends and `args.n` > 1, there will
            # be several states that also need to be updated. Perform the updates
            # in the order in which you encountered the states in the trajectory
            # and during these updates, use the `compute_target_policy(V)` with
            # the up-to-date value of `V`.

            if tau >= 0:
                G = 0
                G += V[states[tau]]
                current_gamma = 1
                current_lambda = 1
                current_rho = 1
                target_policy = compute_target_policy(V)
                for i in range(min(args.n, T-tau)):
                    importance_sampling = target_policy[states[tau+i]][actions[tau+i]] / action_probs[tau+i]
                    if args.vtrace_clip:
                        importance_sampling = min(args.vtrace_clip, importance_sampling)
                    current_rho *= importance_sampling
                    cur_state = states[tau+i]
                    cur_reward = rewards[tau+i]
                    cur_done = dones[tau+i]
                    cur_next_state = next_states[tau+i]
                    dt = (cur_reward + (1-cur_done) * args.gamma * V[cur_next_state] - V[cur_state])
                    G += current_gamma * current_lambda * current_rho * dt
                    current_gamma *= args.gamma
                    if args.trace_lambda:
                        current_lambda *= args.trace_lambda


                V[states[tau]] += args.alpha * (G - V[states[tau]])

            t+= 1
            state = next_state

    return V


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    V = main(args)

    env, R, D, N = create_env(args, report_each=0, evaluate_for=1000)
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            action = argmax_with_tolerance(R[state] + (1 - D[state]) * args.gamma * V[N[state]])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
