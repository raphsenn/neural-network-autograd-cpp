
#include "./NeuralNetwork.h"


int main() {
  
  {
    Matrix<int> X_train = std::vector<std::vector<int>>({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    Matrix<int> y_train = std::vector<std::vector<int>>({{0}, {1}, {1}, {1}});
    NeuralNetwork<float> orGate(std::vector<size_t>({2, 1}), std::vector<Activation>({Activation::step}), 0.1f, InitState::RANDOM);
    return 1;
  }
  return 1;
}