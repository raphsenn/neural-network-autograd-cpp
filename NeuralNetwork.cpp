
#include "./NeuralNetwork.h"

template <typename T>
NeuralNetwork<T>::NeuralNetwork(std::vector<size_t> layers, std::vector<Activation> activation_functions, float learning_rate, InitState state) {
  // Only to avoid warnings. 
  learningRate_ = learning_rate;
  layerSizes_ = layers;
  switch (state) {
    case InitState::RANDOM: break;
    case InitState::ONES: break;
    case InitState::EMPTY: break;
    case InitState::ZERO: break;

  }
}
template class NeuralNetwork<float>;