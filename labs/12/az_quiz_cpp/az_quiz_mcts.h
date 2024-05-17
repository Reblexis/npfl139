#pragma once

#include <array>
#include <functional>
#include <memory>

#include "az_quiz.h"

typedef std::array<float, AZQuiz::ACTIONS> Policy;

typedef std::function<void(const AZQuiz&, Policy&, float&)> Evaluator;

struct MCTNode{
    public:
        std::array<MCTNode*, AZQuiz::ACTIONS> children; // ouch (no gc)
        AZQuiz game;
        float totalValue;
        float prior;
        int visitCount;
        int validChildrenCount;

        MCTNode(float prior){
            this->prior = prior;
            visitCount = 0;
            totalValue = 0;
            validChildrenCount = 0;
            for (int i = 0; i < AZQuiz::ACTIONS; i++) {
                children[i] = nullptr;
            }
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
                        children[i] = new MCTNode(policy[i] / policySum);
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

        pair<int, MCTNode*> selectChild(){
            if(!isEvaluated())
                throw std::runtime_error("Cannot select a child from a node that has not been evaluated");

            float bestValue = -INFINITY;
            pair<int, MCTNode*> bestChild = {-1, nullptr};

            for(int i = 0; i < AZQuiz::ACTIONS; i++){
                if(children[i]){
                    float C = 1.25;
                    float UCBScore = -children[i]->value() + C * children[i]->prior * sqrt(visitCount) / (1 + children[i]->visitCount);
                    if(UCBScore > bestValue){
                        bestValue = UCBScore;
                        bestChild = {i, children[i]};
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

    MCTNode root(0);
    root.evaluate(game, evaluator);
    root.addExplorationNoise(epsilon, alpha);

    for(int i = 0; i < num_simulations; i++){
        MCTNode* node = &root;
        vector<MCTNode*> parents;
        int action = -1;

        while(node->validChildrenCount > 0){
            parents.push_back(node);
            auto [action, node] = node->selectChild();
        }

        if(!node->isEvaluated()){
            AZQuiz game = parents.back()->game;
            game.move(action);
            node->evaluate(std::move(game), evaluator);
        }
        else{
            node->totalValue += -1;
            node->visitCount++;
        }

        float value = node->value();

        std::reverse(parents.begin(), parents.end());
        for(auto parent : parents){
            value = -value;
            parent->totalValue += value;
            parent->visitCount++;
        }
    }

    float totalVisits = 0;
    for(int i = 0; i < AZQuiz::ACTIONS; i++){
        if(root.children[i]){
            totalVisits += root.children[i]->visitCount;
        }
    }
    for(int i = 0; i < AZQuiz::ACTIONS; i++){
        if(root.children[i]){
            policy[i] = static_cast<float>(root.children[i]->visitCount) / totalVisits;
        }
    }
}
