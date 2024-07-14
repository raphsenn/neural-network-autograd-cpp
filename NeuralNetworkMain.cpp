
#include "./NeuralNetwork.h"


int main() {
  
  {
    Matrix<float> X_train = std::vector<std::vector<float>>({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    Matrix<float> y_train = std::vector<std::vector<float>>({{0}, {1}, {1}, {1}});
    NeuralNetwork<float, float> orGate(std::vector<size_t>({2, 64,4, 1}), std::vector<Activation>({Activation::linear, Activation::linear, Activation::linear}), 0.1f, InitState::RANDOM);
    orGate.train(X_train, y_train,1, 0.1f, 100, true);
    orGate.act(X_train).print();
    return 1;
  }
  return 1;
}