
#include "./NeuralNetwork.h"

#include <iostream>

int main() {

  {
    // ________________________________________________________________________
    // Example 1:

    // Simplest Neural Network, no complexity.
    // This neural network learns how to halve a number.
    std::cout << "Learns how to half a number." << std::endl;
    Matrix<float> X = std::vector<std::vector<float>>(
        {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
    Matrix<float> y = std::vector<std::vector<float>>(
        {{0}, {0.5f}, {1.0f}, {1.5f}, {2.0f}, {2.5f}, {3.0f}, {3.5f}});
    NeuralNetwork<float> half(std::vector<size_t>({1, 1}),
                              std::vector<Activation>({Activation::linear}),
                              0.01f, InitState::RANDOM);
    half.train(X, y, 0.1f, 1000, false);
    Matrix<float> X_test = std::vector<std::vector<float>>({{-3},
                                                            {-2},
                                                            {-1},
                                                            {0},
                                                            {1},
                                                            {2},
                                                            {3},
                                                            {4},
                                                            {5},
                                                            {6},
                                                            {7},
                                                            {8},
                                                            {9},
                                                            {10}});
    half.act(X_test).print();
    std::cout << "\n";

    // ________________________________________________________________________
    // Example 2:

    // This neural network learns how to solve the OR-Gate problem.
    std::cout << "Solving the OR-Gate problem." << std::endl;
    Matrix<float> X_train =
        std::vector<std::vector<float>>({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    Matrix<float> y_train_or =
        std::vector<std::vector<float>>({{0}, {1}, {1}, {1}});
    NeuralNetwork<float> orGate(std::vector<size_t>({2, 1}),
                                std::vector<Activation>({Activation::sigmoid}),
                                0.1f, InitState::RANDOM);
    orGate.train(X_train, y_train_or, 0.1f, 10000, false);
    orGate.act(X_train).print();
    std::cout << "\n";

    // ________________________________________________________________________
    // Example 3:

    // This neural network learns how to solve the AND-Gate problem.
    std::cout << "Solving the AND-Gate problem." << std::endl;
    Matrix<float> y_train_and =
        std::vector<std::vector<float>>({{0}, {0}, {0}, {1}});
    NeuralNetwork<float> andGate(std::vector<size_t>({2, 1}),
                                 std::vector<Activation>({Activation::sigmoid}),
                                 0.1f, InitState::RANDOM);
    andGate.train(X_train, y_train_and, 0.1f, 10000, false);
    andGate.act(X_train).print();
    std::cout << "\n";

    // ________________________________________________________________________
    // Example 4:

    // This neural network learns how to solve the XOR-Gate problem.
    std::cout << "Solving the XOR-Gate problem." << std::endl;
    Matrix<float> y_train_xor =
        std::vector<std::vector<float>>({{0}, {1}, {1}, {0}});

    NeuralNetwork<float> xorGate(
        std::vector<size_t>({2, 4, 1}),
        std::vector<Activation>({Activation::sigmoid, Activation::sigmoid}),
        0.88f, InitState::RANDOM);

    xorGate.train(X_train, y_train_xor, 0.1f, 10000, false);
    xorGate.act(X_train).print();
    xorGate.evaluate(X_train, y_train_xor);
    std::cout << "\n";
  }
  return 1;
}