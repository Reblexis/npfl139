#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "az_quiz.h"
#include "az_quiz_mcts.h"
#include "az_quiz_sim_game.h"

namespace py = pybind11;

class AZQuizCpp {
 public:
  template <typename T> using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

  static np_array<float> mcts(np_array<int8_t> board, int8_t to_play, bool randomized,
                              const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& evaluate,
                              int num_simulations, float epsilon, float alpha) {
    if (board.ndim() != 2 || board.shape(0) != AZQuiz::N || board.shape(1) != AZQuiz::N)
      throw std::invalid_argument("The az_quiz_cpp.mcts got a board of incorrect shape");

    AZQuiz game(randomized);
    std::copy_n(board.data(), game.board.size(), game.board.begin());
    game.to_play = to_play;

    Evaluator evaluator = [&evaluate] (const AZQuiz& game, Policy& policy, float& value) {
      auto board = np_array<float>({1, game.N, game.N, game.C});
      game.representation(board.mutable_data());
      auto [np_policy, np_value] = evaluate(board);
      if (np_policy.ndim() != 2 || np_policy.shape(0) != 1 || np_policy.shape(1) != AZQuiz::ACTIONS)
        throw std::invalid_argument("The evaluator given to az_quiz_cpp.mcts returned a policy of incorrect shape");
      if (np_value.ndim() != 1 || np_value.shape(0) != 1)
        throw std::invalid_argument("The evaluator given to az_quiz_cpp.mcts returned a value of incorrect shape");
      std::copy_n(np_policy.data(), policy.size(), policy.begin());
      value = *np_value.data();
    };

    Policy policy;
    ::mcts(game, evaluator, num_simulations, epsilon, alpha, policy);

    return np_array<float>(policy.size(), policy.data());
  }

  static void simulated_games_start(int threads, bool randomized, int num_simulations, int sampling_moves, float epsilon, float alpha) {
    ::simulated_games_start(threads, randomized, num_simulations, sampling_moves, epsilon, alpha);
  }

  static py::list simulated_game(const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& evaluate) {
    BatchEvaluator evaluator = [&evaluate] (const Batch& batch) {
      auto board = np_array<float>({int(batch.size()), AZQuiz::N, AZQuiz::N, AZQuiz::C});
      auto board_data = board.mutable_data();
      for (auto& [game, policy, value] : batch) {
        game->representation(board_data);
        board_data += AZQuiz::N * AZQuiz::N * AZQuiz::C;
      }
      auto [policies, values] = evaluate(board);
      if (policies.ndim() != 2 || policies.shape(0) != int(batch.size()) || policies.shape(1) != AZQuiz::ACTIONS)
        throw std::invalid_argument("The evaluator given to az_quiz_cpp.simulated_game returned a policy of incorrect shape");
      if (values.ndim() != 1 || values.shape(0) != int(batch.size()))
        throw std::invalid_argument("The evaluator given to az_quiz_cpp.simulated_game returned a value of incorrect shape: "+ std::to_string(values.shape(0)) + " dim: " + std::to_string(values.ndim()));
      auto policies_data = policies.data();
      auto values_data = values.data();
      for (auto& [game, policy, value] : batch) {
        std::copy_n(policies_data, policy->size(), policy->begin());
        policies_data += policy->size();
        *value = *values_data++;
      }
    };

    auto history = ::simulated_game(evaluator);

    py::list results{};
    for (auto& [game, policy, value] : *history) {
      auto board = np_array<float>({game.N, game.N, game.C});
      game.representation(board.mutable_data());
      results.append(py::make_tuple(board, np_array<float>(policy.size(), policy.data()), value));
    }
    return results;
  }

  static void simulated_games_stop() {
    ::simulated_games_stop();
  }
};


PYBIND11_MODULE(az_quiz_cpp, m) {
    m.doc() = "AZQuiz C++ Module";

    m.def("mcts", &AZQuizCpp::mcts, "Run a Monte Carlo Tree Search",
          py::arg("board"), py::arg("to_play"), py::arg("randomized"), py::arg("evaluate"),
          py::arg("num_simulations"), py::arg("epsilon"), py::arg("alpha"));
    m.def("simulated_games_start", &AZQuizCpp::simulated_games_start, "Start generating simulated games",
          py::arg("threads"), py::arg("randomized"), py::arg("num_simulations"),
          py::arg("sampling_moves"), py::arg("epsilon"), py::arg("alpha"));
    m.def("simulated_game", &AZQuizCpp::simulated_game, "Get one simulated game", py::arg("evaluate"));
    m.def("simulated_games_stop", &AZQuizCpp::simulated_games_stop, "Shut down the application including the worker threads");
}
