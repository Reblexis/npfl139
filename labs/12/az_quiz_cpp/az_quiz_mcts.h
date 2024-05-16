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
        int validChildrenCount;

    public:
        MCTNode(float prior){
            this->prior = prior;
            visitCount = 0;
            totalValue = 0;
            validChildrenCount = 0;
        }

        float value() const {
            return visitCount > 0 ? totalValue / visitCount : 0;
        }

        bool isEvaluated() const {
            return visitCount > 0;
        }

        void evaluate(AZQuiz&& game, const Evaluator& evaluator){
            if(isEvaluated())
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
                        validChildrenCount++;
                        children[i] = std::make_unique<MCTNode>(policy[i] / policySum);
                    }
                }
            }

            visitCount = 1;
            totalValue = value;
        }

        void addExplorationNoise(float epsilon, float alpha){
            if(!isEvaluated())
                throw std::runtime_error("Cannot add exploration noise to a node that has not been evaluated");

            std::random_device rd;
            std::mt19937 gen(rd());
            std::gamma_distribution<float> gamma(alpha, 1.0);

            std::vector<double> dirichlet(validChildrenCount);
            double sum = 0.0;

            // Generate gamma distributed numbers and compute their sum
            for (auto& value : dirichlet) {
                value = d(gen);
                sum += value;
            }

            // Normalize gamma distributed numbers to get dirichlet distribution
            for (auto& value : dirichlet) {
                value /= sum;
            }

            for(int i = 0; i < AZQuiz::ACTIONS; i++){
                if(children[i]){
                    children[i]->prior = (1 - epsilon) * children[i]->prior + epsilon * dirichlet.back();
                    dirichlet.pop_back();
                }
            }
        }

        MCTNode* selectChild(){
            if(!isEvaluated())
                throw std::runtime_error("Cannot select a child from a node that has not been evaluated");

            float bestValue = -INFINITY;
            MCTNode* bestChild = nullptr;

            for(int i = 0; i < AZQuiz::ACTIONS; i++){
                if(children[i]){
                    float u = prior * sqrt(visitCount) / (1 + children[i]->visitCount);
                    float value = children[i]->value() + u;
                    if(value > bestValue){
                        bestValue = value;
                        bestChild = children[i].get();
                    }
                }
            }

            return bestChild;
        }
};


void mcts(const AZQuiz& game, const Evaluator& evaluator, int num_simulations, float epsilon, float alpha, Policy& policy) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a policy and
  // a value function for the given game.
}
