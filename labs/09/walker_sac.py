#!/usr/bin/env python3
import argparse
import collections
import copy
import random

import gymnasium as gym
import numpy as np
import torch

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="HalfCheetah-v5", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--envs", default=8, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=20, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="walker.model", type=str, help="Model path")
parser.add_argument("--replay_buffer_size", default=1e6, type=int, help="Replay buffer size")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=5e-3, type=float, help="Target network update weight.")
parser.add_argument("--default_alpha", default=0.2, type=float, help="Default alpha.")
parser.add_argument("--autotune_alpha", default=True, action="store_true", help="Autotune alpha.")

USE_WANDB = True

if USE_WANDB:
    import wandb

class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create an actor.
        class Actor(torch.nn.Module):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                # TODO: Create
                # - two hidden layers with `hidden_layer_size` and ReLU activation
                # - a layer for generating means with `env.action_space.shape[0]` units and no activation
                # - a layer for generating sds with `env.action_space.shape[0]` units and `torch.exp` activation
                ...

                self.layer1 = torch.nn.Linear(env.observation_space.shape[0], hidden_layer_size)
                self.relu1 = torch.nn.ReLU()
                self.layer2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
                self.relu2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
                self.relu3 = torch.nn.ReLU()
                self.mean_layer = torch.nn.Linear(hidden_layer_size, env.action_space.shape[0])
                self.sds_layer = torch.nn.Linear(hidden_layer_size, env.action_space.shape[0])

                # Then, create a variable representing a logarithm of alpha, using for example the following:
                self._log_alpha = torch.nn.Parameter(torch.tensor(np.log(args.default_alpha), dtype=torch.float32))
                if USE_WANDB:
                    wandb.log({"log_alpha": self._log_alpha.item()})

                # Finally, create two tensors representing the action scale and offset.
                self.register_buffer("action_scale", torch.as_tensor((env.action_space.high - env.action_space.low) / 2))
                self.register_buffer("action_offset", torch.as_tensor((env.action_space.high + env.action_space.low) / 2))

                self.tune_alpha = args.autotune_alpha

            def forward(self, inputs: torch.Tensor, sample: bool):
                # TODO: Perform the actor computation
                # - First, pass the inputs through the first hidden layer
                #   and then through the second hidden layer.
                # - From these hidden states, compute
                #   - `mus` (the means),
                #   - `sds` (the standard deviations).
                # - Then, create the action distribution using `torch.distributions.Normal`
                #   with the `mus` and `sds`.
                # - We then bijectively modify the distribution so that the actions are
                #   in the given range. Luckily, `torch.distributions.transforms` offers
                #   a class `torch.distributions.TransformedDistribution` than can transform
                #   a distribution by a given transformation. We need to use
                #   - `torch.distributions.transforms.TanhTransform()`
                #     to squash the actions to [-1, 1] range, and then
                #   - `torch.distributions.transforms.AffineTransform(self.action_offset, self.action_scale)`
                #     to scale the action ranges to [low, high].
                #   - To compose these transformations, use
                #     `torch.distributions.transforms.ComposeTransform([t1, t2], cache_size=1)`
                #     with `cache_size=1` parameter for numerical stability.
                #   Note that the `ComposeTransform` can be created already in the constructor
                #   for better performance.
                #   In case you wanted to do this manually, sample from a normal distribution, pass the samples
                #   through the `tanh` and suitable scaling, and then compute the log-prob by using `log_prob`
                #   from the normal distribution and manually accounting for the `tanh` as shown in the slides.
                #   However, the formula from the slides is not numerically stable, for a better variant see
                #   https://github.com/tensorflow/probability/blob/ef1f64a434/tensorflow_probability/python/bijectors/tanh.py#L70-L81
                # - Sample the actions by a `rsample()` call (`sample()` is not differentiable).
                # - Then, compute the log-probabilities of the sampled actions by using `log_prob()`
                #   call. An action is actually a vector, so to be precise, compute for every batch
                #   element a scalar, an average of the log-probabilities of individual action components.
                # - Finally, compute `alpha` as exponentiation of `self._log_alpha`.
                # - Return actions, log_prob, and alpha.
                #
                # Do not forget to support computation without sampling (`sample==False`). You
                # can return for example `torch.tanh(mus) * self.action_scale + self.action_offset`,
                # or you can use for example `sds=1e-7`.

                x = self.relu1(self.layer1(inputs))
                x = self.relu2(self.layer2(x))
                x = self.relu3(self.layer3(x))
                mus = self.mean_layer(x)
                sds = torch.exp(self.sds_layer(x))

                distribution = torch.distributions.Normal(mus, sds)
                final_distribution = torch.distributions.TransformedDistribution(
                    distribution, torch.distributions.transforms.ComposeTransform([
                        torch.distributions.transforms.TanhTransform(),
                        torch.distributions.transforms.AffineTransform(self.action_offset, self.action_scale)
                    ], cache_size=1))

                if sample:
                    actions = final_distribution.rsample()
                else:
                    actions = torch.tanh(mus) * self.action_scale + self.action_offset

                log_prob = final_distribution.log_prob(actions).mean(dim=-1, keepdim=True)
                alpha = torch.exp(self._log_alpha)
                return actions, log_prob, alpha


        # Instantiate the actor as `self._actor`.
        self._actor = Actor(args.hidden_layer_size).apply(wrappers.torch_init_with_xavier_and_zeros).to(self.device)

        class Critic(torch.nn.Module):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                self.layer1 = torch.nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], hidden_layer_size)
                self.relu1 = torch.nn.ReLU()
                self.layer2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
                self.relu2 = torch.nn.ReLU()
                self.layer3 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
                self.relu3 = torch.nn.ReLU()

                self.critic = torch.nn.Linear(hidden_layer_size, 1)

            def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
                x = torch.cat([states, actions], dim=-1)
                x = self.relu1(self.layer1(x))
                x = self.relu2(self.layer2(x))
                x = self.relu3(self.layer3(x))
                return self.layer3(x)

        self.critic1 = Critic(args.hidden_layer_size).apply(wrappers.torch_init_with_xavier_and_zeros).to(self.device)
        self.critic2 = copy.deepcopy(self.critic1)
        self.critic2.apply(wrappers.torch_init_with_xavier_and_zeros)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self._optimizer = torch.optim.Adam(list(self._actor.parameters()) + list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=args.learning_rate)

        # Create MSE loss.
        self._mse_loss = torch.nn.MSELoss()

        self.target_entropy = args.target_entropy * env.action_space.shape[0]

    # Method for performing exponential moving average of weights of the given two modules.
    def update_parameters_by_ema(self, source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)

    def save_actor(self, path: str):
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str):
        self._actor.load_state_dict(torch.load(path, map_location=self.device))

    def save_all(self, path: str):
        self.save_actor("actor_"+path)
        torch.save({
            'actor': self._actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }, path)

    def load_all(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self._actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])

    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        self._optimizer.zero_grad()

        actor_actions, log_prob, alpha = self._actor(states, sample=True)
        critic_values = torch.min(self.critic1(states, actor_actions), self.critic2(states, actor_actions))

        actor_loss = ((alpha.detach() * log_prob)-critic_values).mean()
        actor_loss.backward()

        alpha_loss = (alpha * (-log_prob - self.target_entropy).detach()).mean()
        if self._actor.tune_alpha:
            alpha_loss.backward()

        self.critic1.zero_grad()
        self.critic2.zero_grad()
        critic1_loss = self._mse_loss(self.critic1(states, actions), returns)
        critic2_loss = self._mse_loss(self.critic2(states, actions), returns)
        critic1_loss.backward()
        critic2_loss.backward()

        self._optimizer.step()
        if USE_WANDB:
            wandb.log({"actor_loss": actor_loss.item(), "alpha_loss": alpha_loss.item(), "critic1_loss": critic1_loss.item(), "critic2_loss": critic2_loss.item()})
            wandb.log({"log_alpha": self._actor._log_alpha.item()})

        self.update_parameters_by_ema(self.critic1, self.target_critic1, args.target_tau)
        self.update_parameters_by_ema(self.critic2, self.target_critic2, args.target_tau)

    # Predict actions without sampling.
    @wrappers.typed_torch_function(device, torch.float32)
    def predict_mean_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return predicted actions.
        with torch.no_grad():
            return self._actor(states, sample=False)[0]

    # Predict actions with sampling.
    @wrappers.typed_torch_function(device, torch.float32)
    def predict_sampled_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return sampled actions from the predicted distribution
        with torch.no_grad():
            return self._actor(states, sample=True)[0]

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            actions, log_prob, alpha = self._actor(states, sample=True)
            critic1_values = self.target_critic1(states, actions)
            critic2_values = self.target_critic2(states, actions)
            critic_values = torch.min(critic1_values, critic2_values)
            return critic_values - alpha * log_prob


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            action = network.predict_mean_actions(np.array([state]))[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        random_value = 649526
        network.load_actor(f"{args.env}_{random_value}")
        while True:
            evaluate_episode(True)

    # Create the asynchronous vector environment for training.
    venv = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC)

    # Replay memory of a specified maximum size.
    replay_buffer = wrappers.ReplayBuffer(max_length=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state = venv.reset(seed=args.seed)[0]
    training, autoreset = True, np.zeros(args.envs, dtype=bool)
    steps = 0
    if USE_WANDB:
        wandb.log({"step": steps})

    random_value = random.randint(0, 1000000)

    random_value = 910625
    network.load_all(f"{args.env}_{random_value}")
    wandb.log({"random_value": random_value})
    env_thresholds = {"Pendulum-v1": -180, "InvertedDoublePendulum-v5" : 9200, "BipedalWalker-v3": 210, "BipedalWalkerHardcore-v3": 110, "HalfCheetah-v5": 8200}

    best = 4800
    while training:
        for _ in range(args.evaluate_each):
            # Predict actions by calling `network.predict_sampled_actions`.
            action = network.predict_sampled_actions(state)

            next_state, reward, terminated, truncated, _ = venv.step(action)
            reward = np.where(reward==-100, 0, reward)
            done = terminated | truncated
            for i in range(args.envs):
                if not autoreset[i]:
                    replay_buffer.append(Transition(state[i], action[i], reward[i], done[i], next_state[i]))
            state = next_state
            autoreset = done

            # Training
            if len(replay_buffer) >= 10 * args.batch_size:
                # Randomly uniformly sample transitions from the replay buffer.
                batch = replay_buffer.sample(args.batch_size, np.random)
                states, actions, rewards, dones, next_states = map(np.array, zip(*batch))

                values = network.predict_values(next_states)
                average_value = np.mean(values)
                if USE_WANDB:
                    wandb.log({"average_value": average_value})

                mask = ~dones[:, np.newaxis]
                returns = rewards[:, np.newaxis] + args.gamma * mask * values
                network.train(states, actions, returns)

            steps += args.envs
            if USE_WANDB:
                wandb.log({"steps": steps})

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        print(f"Return after {steps} steps: {np.mean(returns)}")
        wandb.log({"return": np.mean(returns)})

        if np.mean(returns) > best:
            best = np.mean(returns)
            print(f"Best so far: {best}")
            if USE_WANDB:
                wandb.log({"best": best})
            network.save_all(f"{args.env}_{random_value}")

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if USE_WANDB:
        wandb.init(project=args.env)
        wandb.config.update(args)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)