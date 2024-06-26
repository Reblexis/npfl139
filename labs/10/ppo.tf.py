#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf

import multi_collect_environment
import wrappers
multi_collect_environment.register()

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="SingleCollect-v0", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.15, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=32, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=7, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=512, type=int, help="Steps for each worker to perform.")



# TODO: Note that this time we derive the Network directly from `keras.Model`.
# The reason is that the high-level Keras API is useful in PPO, where we need
# to train on an unchanging dataset (generated batches, train for several epochs, ...).
# That means that:
# - we define training in `train_step` method, which the Keras API automatically uses
# - we still provide custom `predict` method, because it is fastest this way
# - loading and saving should be performed using `save_weights` and `load_weights`, so that
#   the `predict` method and the `Network` type is preserved. The `.weights.h5` suffix
#   should be used for the weights file path.
class Network(keras.Model):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace) -> None:
        self._args = args

        # Create a suitable model for the given observation and action spaces.
        inputs = keras.Input(observation_space.shape)

        # TODO: Using a single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a policy with `action_space.n` discrete actions.
        policy = keras.models.Sequential([
            keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            keras.layers.Dense(action_space.n, activation="softmax"),
        ])(inputs)

        # TODO: Using an independent single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a value function estimate. It is best to generate it as a scalar, not
        # a vector of length one, to avoid broadcasting errors later.
        value = keras.models.Sequential([
            keras.layers.Dense(args.hidden_layer_size, activation="relu"),
            keras.layers.Dense(1),
        ])(inputs)

        # Construct the model
        super().__init__(inputs=inputs, outputs=[policy, value])

        # Compile using Adam optimizer with the given learning rate.
        self.compile(optimizer=keras.optimizers.Adam(args.learning_rate))

    # TODO: Define a training method `train_step`, which is automatically used by Keras.
    def train_step(self, data):
        # Unwrap the data. The targets is a dictionary of several tensors, containing keys
        # - "actions"
        # - "action_probs"
        # - "advantages"
        # - "returns"
        states, targets = data
        with tf.GradientTape() as tape:
            # Compute the policy and the value function
            policy, value = self(states, training=True)

            # TODO: Sum the following three losses
            # - the PPO loss, where `self._args.clip_epsilon` is used to clip the probability ratio
            # - the MSE error between the predicted value function and target returns
            # - the entropy regularization with coefficient `self._args.entropy_regularization`.
            #   You can compute it for example using `keras.losses.CategoricalCrossentropy()`
            #   by realizing that entropy can be computed using cross-entropy.
            actions = targets["actions"]
            action_probs = targets["action_probs"]
            advantages = targets["advantages"]
            returns = targets["returns"]

            ratio = tf.reduce_sum(policy * tf.one_hot(actions, policy.shape[-1]), axis=-1) / action_probs
            clipped_ratio = tf.clip_by_value(ratio, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon)
            ppo_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            value_loss = tf.reduce_mean(tf.square(value - returns))
            entropy_loss = -tf.reduce_mean(keras.losses.categorical_crossentropy(policy, policy))

            loss = ppo_loss + value_loss + self._args.entropy_regularization * entropy_loss

        # Perform an optimizer step and return the loss for reporting and visualization.
        self.optimizer.apply(tf.gradients(loss, self.trainable_variables), self.trainable_variables)
        return {"loss": loss}

    @wrappers.raw_typed_tf_function(tf.float32)
    def predict(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env.observation_space, env.action_space, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # TODO: Predict the action using the greedy policy
            action = np.argmax(network.predict(np.array([state]))[0])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    venv = gym.make_vec(env.spec.id, args.envs, gym.VectorizeMode.ASYNC)

    # Training
    state = venv.reset(seed=args.seed)[0]
    training = True
    iteration = 0
    while training:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[args.worker_steps, args.envs]`.

        states = np.zeros((args.worker_steps, args.envs, env.observation_space.shape[0]), dtype=np.float32)
        actions = np.zeros((args.worker_steps, args.envs, 1), dtype=np.int32)
        action_probs, rewards, dones = [np.zeros((args.worker_steps, args.envs, 1), dtype=np.float32) for _ in range(3)]
        values = np.zeros((args.worker_steps + 1, args.envs, 1), dtype=np.float32)
        values[0] = network.predict(state)[1]

        for t in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.envs` actions, each
            # sampled from the corresponding policy generated by the `network.predict`
            # executed on the vector `state`.

            probs, values[t + 1] = network.predict(state)
            action = np.array([np.random.choice(env.action_space.n, p=probs[i]) for i in range(args.envs)])[:, np.newaxis]

            # Perform the step
            next_state, reward, terminated, truncated, _ = venv.step(action[:, 0])
            done = terminated | truncated

            states[t] = state
            actions[t] = action
            corresponding_probs = np.array([probs[i][action[i]] for i in range(len(probs))])
            action_probs[t] = corresponding_probs
            rewards[t] = reward[:, np.newaxis]
            dones[t] = done[:, np.newaxis]


            state = next_state

        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to process episodes of individual workers independently, and note that
        # each worker might have generated multiple episodes, the last one probably unfinished.

        advantages, returns = np.zeros_like(rewards), np.zeros_like(rewards)
        for i in range(args.envs):
            g = 0
            for t in reversed(range(args.worker_steps)):
                td_error = rewards[t][i] + (1 - dones[t][i]) * args.gamma * values[t + 1][i] - values[t][i]
                g = td_error + (1-dones[t][i]) * args.gamma * args.trace_lambda * g
                advantages[t][i] = g
                returns[t][i] = g + values[t][i]


        # Train using the Keras API.
        # - The below code assumes that the first two dimensions of the used quantities are
        #   `[args.worker_steps, args.envs]` and concatenates them together.
        # - We do not log the training by passing `verbose=0`; feel free to change it.
        network.fit(
            np.reshape(states, (-1, states.shape[-1])),
            {"actions": np.reshape(actions, (-1, 1)),
             "action_probs": np.reshape(action_probs, (-1, 1)),
             "advantages": np.reshape(advantages, (-1, 1)),
             "returns": np.reshape(returns, (-1, 1))},
            batch_size=args.batch_size, epochs=args.epochs, verbose=0,
        )

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            returns = [evaluate_episode() for _ in range(args.evaluate_for)]

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
