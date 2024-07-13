
#include <algorithm>

#include "./NeuralNetwork.h"

template <typename S, typename T>
NeuralNetwork<S, T>::NeuralNetwork(std::vector<size_t> layers, std::vector<Activation> activation_functions, float learning_rate, InitState state) {
  layerSizes_ = layers;
  numLayers_ = layers.size();
  learningRate_ = learning_rate;
  
  // Create weights and biases. 
  for (size_t i = 0; i < layers.size() - 1; ++i) {
    weights_.push_back(Matrix<T>(layers[i], layers[i+1], state));
    biases_.push_back(Matrix<T>(1, layers[i+1], state));
  }

  for (const auto& act : activation_functions) {
    // Not pretty i guess.
    // https://en.cppreference.com/w/cpp/utility/functional/function
    switch (act) { 
      case Activation::linear: 
        activationFunctions_.push_back([](Matrix<T>& X) {return linear(X); }); 
        activationFunctionDerivatives_.push_back([](Matrix<T>& X) {return linear_derivative(X); }); break;
      case Activation::relu: 
        activationFunctions_.push_back([](Matrix<T>& X) {return relu(X); }); 
        activationFunctionDerivatives_.push_back([](Matrix<T>& X) {return relu_derivative(X); }); break;
      case Activation::step: 
        activationFunctions_.push_back([](Matrix<T>& X) {return step(X); }); 
        activationFunctionDerivatives_.push_back([](Matrix<T>& X) {return step_derivative(X); }); break;
    }
  }
}

template <typename S, typename T>
Matrix<T> NeuralNetwork<S, T>::forward(Matrix<T> X) {
  A_ = {X};
  Z_.clear();

  // Forward pass.
  for (size_t i = 0; i < numLayers_ - 1; ++i) {
    // A_.back().print();
    // weights_[i].print();
    // biases_[i].print();
    Matrix<T> z = A_.back().dot(weights_[i]).add(biases_[i]);
    Z_.push_back(z);
    A_.push_back(activationFunctions_[i](z));
  }
  return A_.back();
}

template <typename S, typename T>
void NeuralNetwork<S, T>::backward(Matrix<T> X, Matrix<T> y) {
  Matrix<T> output_error = A_.back().sub(y);
  // std::cout << "\n"; 
  
  // output_error.print();
  // activationFunctionDerivatives_.back()(A_.back()).print();
  
  Matrix<T> delta = output_error.dotElementWise(activationFunctionDerivatives_.back()(A_.back()));
  // std::cout << "this one worked" << std::endl; 

  std::vector<Matrix<T>> deltas = {delta};

  for (size_t i = numLayers_ - 2; i > 0;  --i) {
    delta = deltas.back().dot(weights_[i].transpose()).dotElementWise(activationFunctionDerivatives_[i](A_[i]));
    deltas.push_back(delta);
  }
  // std::cout << "this one worked too" << std::endl; 
  std::reverse(deltas.begin(), deltas.end());

  for (size_t i = 0; i < numLayers_ - 1; ++i) {
    weights_[i].add(A_[i].transpose().dot(deltas[i]));
    biases_[i].add(deltas[i].sum());
  }


}

template <typename S, typename T>
void NeuralNetwork<S, T>::train(Matrix<T> X, Matrix<T> y, size_t batch_size, float learning_rate, int epochs, bool verbose) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // std::cout << "Starting forward pass." << std::endl; 
    Matrix<T> output = forward(X);
    // std::cout << "Forward pass done." << std::endl; 
    // std::cout << "Starting backward pass." << std::endl; 
    backward(X, y);
    // std::cout << "Backward pass done." << std::endl; 
    if (verbose) { std::cout << "Epoch: " << epoch << std::endl; }
  }
}

template <typename S, typename T>
Matrix<T> NeuralNetwork<S, T>::act(Matrix<T> X) {
  return forward(X);
}

template class NeuralNetwork<int, float>;
template class NeuralNetwork<float, float>;