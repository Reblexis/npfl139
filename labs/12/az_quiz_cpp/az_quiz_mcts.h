#pragma once

#include <array>
#include <functional>
#include <memory>

#include "az_quiz.h"

typedef std::array<float, AZQuiz::ACTIONS> Policy;

typedef std::function<void(const AZQuiz&, Policy&, float&)> Evaluator;

class MCTNode{
    private:
        float prior;
        AZQuiz game;
        std::array<std::unique_ptr<MCTNode>, AZQuiz::ACTIONS> children;
        int visitCount;
        float totalValue;

    public:
        MCTNode(float prior){
            this->prior = prior;
            visitCount = 0;
            totalValue = 0;
        }

        float value() const {
            return visitCount > 0 ? totalValue / visitCount : 0;
        }

        bool isEvaluated() const {
            return visitCount > 0;
        }

        void evaluate(AZQuiz&& game, const Evaluator& evaluator){
            if(visitCount>0)
                throw std::runtime_error("Cannot evaluate a node more than once");

            this->game = std::move(game);

            float value = 0
            if(game.winner >= 0)
                value = -1;
            else{
                Policy policy;
                evaluator(game, policy, value);
                float policySum = 0;
                for(int i = 0; i < AZQuiz::ACTIONS; i++){
                    if(game.valid(i)){
                        policySum += policy[i];
                    }
                }
                for(int i = 0; i < AZQuiz::ACTIONS; i++){
                    if(game.valid(i)){
                        children[i] = std::make_unique<MCTNode>(policy[i] / policySum);
                    }
                }
            }

            visitCount = 1;
            totalValue = value;
        }

        void addExplorationNoise(float epsilon, float alpha){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::gamma_distribution<float> gamma(alpha, 1.0);

            double sum = 0.0;
        }
};


void mcts(const AZQuiz& game, const Evaluator& evaluator, int num_simulations, float epsilon, float alpha, Policy& policy) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a policy and
  // a value function for the given game.
}
