#!/usr/bin/env python3
import argparse
import collections
import os
import time

import numpy as np
import torch

from pisqorky import Pisqorky
import pisqorky_evaluator
import pisqorky_player_heuristic
import wrappers

import pisqorky_cpp

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=64, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="az_quiz.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=100, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--sampling_moves", default=10, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=32, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=1, type=int, help="Update steps in every iteration.")
parser.add_argument("--window_length", default=100_000, type=int, help="Replay buffer max length.")


#########
# Agent #
#########

class Network(torch.nn.Module):
    def __init__(self):
        # A possible architecture known to work consists of
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
        # - a policy head, which first uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens the representation, and finally uses a dense layer with softmax
        #   activation to produce the policy,
        # - a value head, which again uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens, and produces expected return using an output dense layer with
        #   `tanh` activation.
        super().__init__()

        self.conv1 = torch.nn.Conv2d(4, 15, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(15, 15, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(15, 15, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(15, 15, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(15, 20, kernel_size=3, padding=1)

        self.policy_conv = torch.nn.Conv2d(20, 2, kernel_size=3, padding=1)
        self.policy_dense = torch.nn.Linear(2 * 7 * 7, 225)

        self.value_conv = torch.nn.Conv2d(20, 2, kernel_size=3, padding=1)
        self.value_dense = torch.nn.Linear(2 * 7 * 7, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))

        policy = torch.nn.functional.relu(self.policy_conv(x))
        policy = torch.flatten(policy, 1)
        policy = torch.nn.functional.softmax(self.policy_dense(policy), dim=-1)

        value = torch.nn.functional.relu(self.value_conv(x))
        value = torch.flatten(value, 1)
        value = torch.tanh(self.value_dense(value))

        return policy, value


class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, args: argparse.Namespace):
        self._model = Network().to(self.device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        # A static method returning a new Agent loaded from the given path.
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @wrappers.typed_torch_function(device, torch.float32, torch.float32, torch.float32, via_np=True)
    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> None:
        # TODO: Train the model based on given boards, target policies and target values.
        policy, values = self._model(boards)

        value_loss = torch.nn.functional.mse_loss(values.squeeze(), target_values)
        policy_loss = torch.nn.functional.cross_entropy(policy, target_policies)

        loss = value_loss + policy_loss

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @wrappers.typed_torch_function(device, torch.float32, via_np=True)
    def predict(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function.
        with torch.no_grad():
            return self._model(boards)

    def board(self, game: AZQuiz) -> np.ndarray:
        # TODO: Generate the boards from the current `AZQuiz` game.
        #
        # The `game.board` returns a board representation, but you also need to
        # somehow indicate who is the current player. You can either
        # - change the game so that the current player is always the same one
        #   (i.e., always 0 or always 1; `swap_players` of `AZQuiz.clone` might come handy);
        # - indicate the current player by adding channels to the representation.

        if game.to_play == 1:
            game = game.clone(swap_players=True)

        board = game.board

        # move channel to front
        board = np.moveaxis(board, -1, 0)

        return board

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.moveaxis(x, -1, 1)
        return self.predict(x)


############
# Training #
############

def train(args: argparse.Namespace) -> Agent:
    # Perform training
    agent = Agent(args)
    replay_buffer = wrappers.ReplayBuffer(max_length=args.window_length)

    iteration = 0
    training = True
    best_score = 0
    while training:
        iteration += 1

        pisqorky_cpp.simulated_games_start(args.sim_games, False, args.num_simulations, args.sampling_moves, args.epsilon, args.alpha)
        # Generate simulated games
        for _ in range(args.sim_games):
            game = pisqorky_cpp.simulated_game(agent)
            replay_buffer.extend(game)

            # If required, show the generated game, as 8 very long lines showing
            # all encountered boards, each field showing as
            # - `XX` for the fields belonging to player 0,
            # - `..` for the fields belonging to player 1,
            # - percentage of visit counts for valid actions.
            if args.show_sim_games:
                log = [[] for _ in range(8)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append("Move {}, result {}".format(i, outcome).center(28))
                    action = 0
                    for row in range(7):
                        log[1 + row].append("  " * (6 - row))
                        for col in range(row + 1):
                            log[1 + row].append(
                                " XX " if board[row, col, 0] else
                                " .. " if board[row, col, 1] else
                                "{:>3.0f} ".format(policy[action] * 100))
                            action += 1
                        log[1 + row].append("  " * (6 - row))
                print(*["".join(line) for line in log], sep="\n")

        pisqorky_cpp.simulated_games_stop()
        print("Training...")
        # Train
        for _ in range(args.train_for):
            # TODO: Perform training by sampling an `args.batch_size` of positions
            # from the `replay_buffer` and running `agent.train` on them.
            sample = replay_buffer.sample(args.batch_size) # sample is a list of tuples (board, policy, outcome)
            boards = np.array([sample[i][0] for i in range(args.batch_size)])
            boards = np.moveaxis(boards, -1, 1)
            policies = np.array([sample[i][1] for i in range(args.batch_size)])
            outcomes = np.array([sample[i][2] for i in range(args.batch_size)])

            agent.train(boards, policies, outcomes)

        # Evaluate
        if iteration % args.evaluate_each == 0:
            print("Evaluating...")
            # Run an evaluation on 2*56 games versus the simple heuristics,
            # using the `Player` instance defined below.
            # For speed, the implementation does not use MCTS during evaluation,
            # but you can of course change it so that it does.
            score = pisqorky_evaluator.evaluate(
                [Player(agent, argparse.Namespace(num_simulations=0)),
                 pisqorky_player_heuristic.Player(seed=args.seed)],
                games=28*10, randomized=False, first_chosen=True, render=False, verbose=False)
            if score > best_score:
                agent.save(args.model_path)
                best_score = score
            print("Evaluation after iteration {}: {:.1f}%".format(iteration, 100 * score), flush=True)

    return agent


#####################
# Evaluation Player #
#####################
class Player:
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: Pisqorky) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # TODO: If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            policy = self.agent.predict(self.agent.board(game)[np.newaxis])[0][0]
        else:
            # TODO: Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            policy = np.zeros(225)
            policy = pisqorky_cpp.mcts(game, self.agent, self.args.num_simulations, self.args.epsilon, self.args.alpha, policy)

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Perform training
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    pisqorky_evaluator.evaluate(
        [player, pisqorky_player_heuristic.Player(seed=args.seed)],
        games=56, randomized=False, first_chosen=False, render=False, verbose=True,
    )
