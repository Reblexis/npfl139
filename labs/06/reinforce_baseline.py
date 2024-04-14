#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch
#import wandb

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")

class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self.policy_model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
        ).to(self.device)

        self.value_model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=args.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=args.learning_rate)

        self.policy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.value_loss = torch.nn.MSELoss()

        self.policy_model.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.value_model.apply(wrappers.torch_init_with_xavier_and_zeros)

    @wrappers.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the policy model, using `returns - predicted_baseline` as
        #   advantage estimate
        # - train the baseline model to predict `returns`

        self.policy_model.train()
        self.policy_optimizer.zero_grad()

        self.value_model.train()
        self.value_optimizer.zero_grad()

        values = self.value_model(states).squeeze()
        values_no_grad = values.detach()
        value_loss = self.value_loss(values, returns)
        value_loss.backward()

        logits = self.policy_model(states)
        policy_loss = self.policy_loss(logits, actions) @ (returns - values_no_grad)
        policy_loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    @wrappers.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self.policy_model.eval()
        with torch.no_grad():
            logits = self.policy_model(states)
            policy = torch.nn.functional.softmax(logits, dim=-1)
            return policy


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    scores = []
    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            score = 0
            while not done:
                # TODO: Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `network.predict` and current `state`.
                action = np.random.choice(env.action_space.n, p=network.predict(state))

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                score += reward

                state = next_state

            scores.append(score)
            # TODO: Compute returns by summing rewards (with discounting)
            returns = [0] * len(rewards)
            returns[-1] = rewards[-1]
            for i in reversed(range(len(rewards) - 1)):
                returns[i] = rewards[i] + args.gamma * returns[i + 1]

            # TODO: Add states, actions and returns to the training batch
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        avg_score = sum(scores[-10:]) / 10
        #wandb.log({"avg_score": avg_score})
        #wandb.log({"episode": env.episode})
        if avg_score > 495:
            break

        # TODO: Train using the generated batch.
        network.train(batch_states, batch_actions, batch_returns)


        # TODO(reinforce): Train using the generated batch.

    """
    final_rewards = []
    for _ in range(100):
        state, done = env.reset(start_evaluation=False)[0], False
        rewards = []
        while not done:
            # TODO(reinforce): Choose a greedy action
            action = np.argmax(network.predict(state))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
        final_rewards.append(sum(rewards))
    """


    #wandb.log({"final_reward": sum(final_rewards) / 100})

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(reinforce): Choose a greedy action
            action = np.argmax(network.predict(state))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)

"""
    wandb.init(
        project="reinforce_cartpole",
        config=vars(args),
    )
"""

