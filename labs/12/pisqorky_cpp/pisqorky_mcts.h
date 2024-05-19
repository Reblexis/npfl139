#pragma once

#include <array>
#include <functional>

#include "pisqorky.h"

typedef std::array<float, Pisqorky::ACTIONS> Policy;

typedef std::function<void(const Pisqorky&, Policy&, float&)> Evaluator;

pid_t get_thread_id() {
    return syscall(SYS_gettid);
}

class MCTNode {
public:
    std::array<std::unique_ptr<MCTNode>, AZQuiz::ACTIONS> children;
    AZQuiz game;
    float totalValue=0.0f;
    float prior=0.0f;
    int visitCount=0;
    int validChildrenCount=0;

    explicit MCTNode(float prior)
        : prior(prior), visitCount(0), totalValue(0), validChildrenCount(0) {}

    float value() const {
        return visitCount > 0 ? totalValue / visitCount : 0;
    }

    bool isEvaluated() const {
        return visitCount > 0;
    }

    void evaluate(AZQuiz game, const Evaluator& evaluator) {
        if (isEvaluated())
            throw std::runtime_error("Cannot evaluate a node more than once");

        this->game = std::move(game);

        float value = 0;
        if (this->game.winner >= 0)
            value = (game.winner==2? 0 : -1);
        else {
            Policy policy;
            pid_t pid = get_thread_id();
            std::cerr<<"calling evaluator "<<pid<<std::endl;
            evaluator(this->game, policy, value);
            std::cerr<<"evaluator finished "<<pid<<std::endl;
            float policySum = 0.0;
            for (int i = 0; i < AZQuiz::ACTIONS; i++) {
                if (this->game.valid(i)) {
                    policySum += policy[i];
                }
            }
            for (int i = 0; i < AZQuiz::ACTIONS; i++) {
                if (this->game.valid(i)) {
                    validChildrenCount++;
                    children[i] = std::make_unique<MCTNode>(policy[i] / policySum);
                }
            }
        }

        visitCount = 1;
        totalValue = value;
    }

    void addExplorationNoise(float epsilon, float alpha) {
        if (!isEvaluated())
            throw std::runtime_error("Cannot add exploration noise to a node that has not been evaluated");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::gamma_distribution<float> gamma(alpha, 1.0);

        std::vector<double> dirichlet(validChildrenCount);
        double sum = 0.0;

        for (auto& value : dirichlet) {
            value = gamma(gen);
            sum += value;
        }

        for (auto& value : dirichlet) {
            value /= sum;
        }

        auto dirIt = dirichlet.begin();
        for (auto& child : children) {
            if (child) {
                child->prior = (1 - epsilon) * child->prior + epsilon * (*dirIt++);
            }
        }
    }

    std::pair<int, MCTNode*> selectChild() {
        if (!isEvaluated())
            throw std::runtime_error("Cannot select a child from a node that has not been evaluated");

        float bestValue = -std::numeric_limits<float>::infinity();
        std::pair<int, MCTNode*> bestChild = {-1, nullptr};

        for (int i = 0; i < AZQuiz::ACTIONS; i++) {
            if (children[i]) {
                float C = 1.25f;
                float UCBScore = -children[i]->value() + C * children[i]->prior * sqrt(static_cast<float>(visitCount)) / (1.0f + children[i]->visitCount);
                if (UCBScore > bestValue) {
                    bestValue = UCBScore;
                    bestChild = {i, children[i].get()};
                }
            }
        }

        return bestChild;
    }
};

void mcts(const Pisqorky& game, const Evaluator& evaluator, int num_simulations, float epsilon, float alpha, Policy& policy) {
    MCTNode root(0.0f);
    root.evaluate(game, evaluator);
    root.addExplorationNoise(epsilon, alpha);

    for (int i = 0; i < num_simulations; i++) {
        MCTNode* node = &root;
        std::vector<MCTNode*> path;

        int action = -1;
        while (node->validChildrenCount > 0) {
            path.push_back(node);
            auto child = node->selectChild();
            action = child.first;
            if(child.second == nullptr)
                throw std::runtime_error("Child is nullptr");
            node = child.second;
        }

        if (!node->isEvaluated()) {
            AZQuiz nextGame = path.back()->game;
            nextGame.move(action);
            node->evaluate(nextGame, evaluator);
        } else {
            node->totalValue += (node->game.winner == 2 ? 0 : -1);
            node->visitCount++;
        }

        float value = node->value();
        std::reverse(path.begin(), path.end());
        for (auto parent : path) {
            value = -value;
            parent->totalValue += value;
            parent->visitCount++;
        }
    }

    float totalVisits = 0.0f;
    for (const auto& child : root.children) {
        if (child) {
            totalVisits += child->visitCount;
        }
    }
    for (int i = 0; i < AZQuiz::ACTIONS; i++) {
        if (root.children[i]) {
            policy[i] = static_cast<float>(root.children[i]->visitCount) / totalVisits;
        }
    }
}
