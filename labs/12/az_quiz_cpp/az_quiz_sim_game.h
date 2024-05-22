#pragma once

#include <cmath>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <random>

#include "az_quiz.h"
#include "az_quiz_mcts.h"

typedef std::vector<std::tuple<AZQuiz, Policy, float>> History;

typedef std::vector<std::tuple<const AZQuiz*, Policy*, float*>> Batch;

typedef std::function<void(const Batch&)> BatchEvaluator;

std::mutex worker_mutex;
std::condition_variable worker_cv;
Batch worker_queue;
size_t worker_queue_limit;

bool worker_shutdown;
class worker_shutdown_exception : public std::exception {};

std::mutex processor_mutex;
std::condition_variable processor_cv;
std::vector<std::unique_ptr<Batch>> processor_queue;
std::vector<std::unique_ptr<History>> processor_result;

void worker_evaluator(const AZQuiz& game, Policy& policy, float& value) {
  std::unique_lock worker_lock{worker_mutex};

  value = INFINITY;
  worker_queue.emplace_back(&game, &policy, &value);
  if (worker_queue.size() == worker_queue_limit) {
    auto batch = std::make_unique<Batch>(worker_queue);
    worker_queue.clear();
    {
      std::unique_lock processor_lock{processor_mutex};
      processor_queue.push_back(std::move(batch));
    }
    processor_cv.notify_one();
  }

  if (!worker_shutdown)
    worker_cv.wait(worker_lock, [&value]{return std::isfinite(value) || worker_shutdown;});
  if (worker_shutdown) throw worker_shutdown_exception();
}

void worker_thread(bool randomized, int num_simulations, int sampling_moves, float epsilon, float alpha) try {
  while (true) {
    auto history = std::make_unique<History>();

    AZQuiz game(randomized);
    int currentMove = 0;

    std::vector<AZQuiz> boards;
    std::vector<Policy> policies;
    std::vector<float> outcomes;

    while(game.winner < 0){
        Policy policy;
        mcts(game, worker_evaluator, num_simulations, epsilon, alpha, policy);

        boards.push_back(game);
        policies.push_back(policy);

        int action = -1;
        if(currentMove < sampling_moves){
            // Sample a move according to the policy distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(policy.begin(), policy.end());
            action = dist(gen);
        }
        else{
            // Argmax
            action = std::distance(policy.begin(), std::max_element(policy.begin(), policy.end()));
        }

        game.move(action);
        currentMove++;
    }

    float perspectiveGameOutcome = 1;
    for(int i = 0; i < boards.size(); i++){
        outcomes.push_back(perspectiveGameOutcome);
        perspectiveGameOutcome *= -1;
    }

    std::reverse(boards.begin(), boards.end());

    for(int i = 0; i < boards.size(); i++){
        history->emplace_back(boards[i], policies[i], outcomes[i]);
    }

    // Once the whole game is finished, we pass it to processor to return it.
    {
      std::unique_lock processor_lock{processor_mutex};
      processor_result.push_back(std::move(history));
    }
    processor_cv.notify_one();
  }
} catch (worker_shutdown_exception&) {
  return;
}

void simulated_games_start(int threads, bool randomized, int num_simulations, int sampling_moves, float epsilon, float alpha) {
  worker_shutdown = false;
  worker_queue_limit = threads;
  for (int thread = 0; thread < threads; thread++)
    std::thread(worker_thread, randomized, num_simulations, sampling_moves, epsilon, alpha).detach();
}

std::unique_ptr<History> simulated_game(BatchEvaluator& evaluator) {
  while (true) {
    std::unique_ptr<Batch> batch;
    {
      std::unique_lock processor_lock{processor_mutex};
      processor_cv.wait(processor_lock, []{return processor_result.size() || processor_queue.size();});
      if (processor_result.size()) {
        auto result = std::move(processor_result.back());
        processor_result.pop_back();
        return result;
      }

      batch = std::move(processor_queue.back());
      processor_queue.pop_back();
    }

    evaluator(*batch);

    {
      std::unique_lock worker_lock{worker_mutex};
    }
    worker_cv.notify_all();
  }
}

void simulated_games_stop() {
  std::unique_lock worker_lock{worker_mutex};
  worker_shutdown = true;
  worker_cv.notify_all();
}
