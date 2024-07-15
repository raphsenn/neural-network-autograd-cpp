
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
      case Activation::sigmoid: 
        activationFunctions_.push_back([](Matrix<T>& X) {return sigmoid(X); }); 
        activationFunctionDerivatives_.push_back([](Matrix<T>& X) {return sigmoid_derivative(X); }); break;
    
    }
  }
}

// ____________________________________________________________________________
// Forward propagation:
template <typename S, typename T>
Matrix<T> NeuralNetwork<S, T>::forward(Matrix<T> X) {
  
  // Pre-active values (weighted sums).
  Z_.clear();
  
  // Activations (of weighted sums).
  A_.clear();
  A_.push_back(X);

  // Forward propagation.
  // In a nutshell:
  // Z:
  // Z_[0] = dot(X, W[0]) + BIAS[0]
  // Z_[1] = dot(ACT_0(dot(X, W[0]) + BIAS[0]), W[1]) + BIAS[1]
  // Z_[2] = dot(ACT_1(dot(ACT_0(dot(X, W[0]) + BIAS[0]), W[1]) + BIAS[1]), W[2]) + BIAS[2
  //
  // A:
  // A_[0] = X
  // A_[1] = ACT_0(dot(X, W[0]) + BIAS[0])
  // A_[2] = ACT_1(dot(ACT_0(dot(X, W[0]) + BIAS[0]), W[1]) + BIAS[1])
  // A_[n] = ACT_n(dot(... (ACT_1(dot(ACT_0(dot(X, W[0]) + BIAS[0]), W[1]) + BIAS[1]) ...W[n]) + BIAS[n]) 

  for (size_t i = 0; i < numLayers_ - 1; ++i) {
    
    // Weighted sums:
    // Calculates: dot(A[i], W[i]) + BIAS[i]
    Matrix<T> z = dot(A_.back(), weights_[i]);
    z = add(z, biases_[i]);
    Z_.push_back(z);

    // Activation of weighted sums: 
    // Calculates: activate(dot(A[i], W[i]) + BIAS)
    Matrix<T> a = activationFunctions_[i](z); 
    A_.push_back(a);
  
  }
  
  // Last element of A_ is output.
  return A_.back();
}

// ____________________________________________________________________________
// Backpropagation:
template <typename S, typename T>
void NeuralNetwork<S, T>::backward(Matrix<T> X, Matrix<T> y) {

  // Backpropagation in a nutshell.
  //
  // 1. Calculate output error:
  // output error = output - y
  //
  // 2. Calculate delta for each layer by propagating the error backwards through the network.

  // Calculate output error.
  // Calculates: output - labels = error
  Matrix<T> output_error = sub(y, A_.back());
  
  // Calculate the derivative of the activation function at the output layer. 
  Matrix<T> activation_derivative = activationFunctionDerivatives_.back()(A_.back()); 
  
  // Compute delta for the output layer using element-wise multiplication of error and activation derivative.
  Matrix<T> delta = dotElementWise(output_error, activation_derivative);

  std::vector<Matrix<T>> deltas;
  deltas.push_back(delta);

  for (size_t i = numLayers_ - 2; i > 0;  --i) {
    Matrix<T> W = std::move(weights_[i].transpose2()); 
    delta = dot(deltas.back(), W);

    activation_derivative = activationFunctionDerivatives_[i](A_[i]); 
    delta = dotElementWise(delta, activation_derivative); 
    deltas.push_back(delta);
  }
  std::reverse(deltas.begin(), deltas.end());

  // Update weights and biases.
  for (size_t i = 0; i < numLayers_ - 1; ++i) {
    
    Matrix<T> A_i_transposed = A_[i].transpose2();
    Matrix<T> dW = dot(A_i_transposed, deltas[i]);
    // Matrix<T> dW = dot(A_[i].transpose(), deltas[i]);
    dW = dW.scalMul(learningRate_);
    Matrix<T> dB = deltas[i].sum(1).scalMul(learningRate_); 
    
    // weights_[i] = add(weights_[i], dW);
    weights_[i].add(dW);
    // biases_[i].add(deltas[i].sum(1).scalMul(learningRate_));
    biases_[i] = add(biases_[i], dB); 
  }
}

template <typename S, typename T>
void NeuralNetwork<S, T>::train(Matrix<T> X, Matrix<T> y, size_t batch_size, float learning_rate, int epochs, bool verbose) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    Matrix<T> output = forward(X);
    backward(X, y);
    if (verbose) { 
      if (epoch % 100 == 0) {std::cout << "Epoch: " << epoch << std::endl; }
    }
  }
  std::cout << "Finished, learning rate = " << learningRate_ << std::endl;
}

template <typename S, typename T>
Matrix<T> NeuralNetwork<S, T>::act(Matrix<T> X) {
  return forward(X);
}

template class NeuralNetwork<int, float>;
template class NeuralNetwork<float, float>;