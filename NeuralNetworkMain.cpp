
#include "./NeuralNetwork.h"
#include <iostream>

int main() {
  
  {
    std::cout << "or" << std::endl;
    Matrix<float> X_train = std::vector<std::vector<float>>({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    Matrix<float> y_train_or = std::vector<std::vector<float>>({{0}, {1}, {1}, {1}});
    NeuralNetwork<float, float> orGate(std::vector<size_t>({2, 4, 1}), std::vector<Activation>({Activation::sigmoid, Activation::sigmoid}), 0.1f, InitState::RANDOM);
    orGate.train(X_train, y_train_or,1, 0.1f, 10000, false);
    orGate.act(X_train).print();
  
    std::cout << "and" << std::endl;
    Matrix<float> y_train_and = std::vector<std::vector<float>>({{0}, {0}, {0}, {1}});
    NeuralNetwork<float, float> andGate(std::vector<size_t>({2, 4, 1}), std::vector<Activation>({Activation::sigmoid, Activation::sigmoid}), 0.01f, InitState::RANDOM);
    andGate.train(X_train, y_train_and,1, 0.1f, 10000, false);
    andGate.act(X_train).print();
  
    std::cout << "xor" << std::endl;
    Matrix<float> y_train_xor = std::vector<std::vector<float>>({{0}, {1}, {1}, {0}});
    NeuralNetwork<float, float> xorGate(std::vector<size_t>({2, 4, 1}), std::vector<Activation>({Activation::sigmoid, Activation::sigmoid}), 0.01f, InitState::RANDOM);
    xorGate.train(X_train, y_train_xor,1, 0.1f, 10000, false);
    xorGate.act(X_train).print();
  
  }
  
  return 1;
}