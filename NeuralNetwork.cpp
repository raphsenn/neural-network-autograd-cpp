
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
  A_.clear();
  A_.push_back(X);
  Z_.clear();

  // Forward propagation.
  for (size_t i = 0; i < numLayers_ - 1; ++i) {
    
    // Calculates: matmul(A[i], W[i]) + BIAS[i] for each layer i
    Matrix<T> z = dot(A_.back(), weights_[i]);
    z = add(z, biases_[i]);
    // Matrix<T> z = A_.back().dot(weights_[i]).add(biases_[i]);
    Z_.push_back(z);
    
    // Calculates: activate(matmul(A[i], W[i]) + BIAS)
    activationFunctions_[i](z); 
    A_.push_back(z);
  }
  // Last element of A_ is output.
  return A_.back();
}

template <typename S, typename T>
void NeuralNetwork<S, T>::backward(Matrix<T> X, Matrix<T> y) {
  // Calculate output error.
  // Calculates: output - labels = error 
  A_.back().print();
  y.print(); 
  Matrix<T> output_error = sub(A_.back(), y);

  // Matrix<T> output_error = A_.back().sub(y);
  
  // Matrix<T> delta = output_error.dotElementWise(activationFunctionDerivatives_.back()(A_.back()));
  Matrix<T> activation_derivative = activationFunctionDerivatives_.back()(A_.back()); 
  Matrix<T> delta = dotElementWise(output_error, activation_derivative);

  std::vector<Matrix<T>> deltas;
  deltas.push_back(delta);
  // delta.print();

  for (size_t i = numLayers_ - 2; i > 0;  --i) {
    // delta = deltas.back().dot(weights_[i].transpose()).dotElementWise(activationFunctionDerivatives_[i](A_[i]));
    delta = dot(deltas.back(), weights_[i].transpose());
    weights_[i].transpose();

    activation_derivative = activationFunctionDerivatives_[i](A_[i]); 
    delta = dotElementWise(delta, activation_derivative); 
    // delta.dotElementWise(activationFunctionDerivatives_[i](A_[i]));
    deltas.push_back(delta);
  }
  std::reverse(deltas.begin(), deltas.end());

  for (size_t i = 0; i < numLayers_ - 1; ++i) {
    std::cout << "Got this 1, iter: "<< i << std::endl;
    
    // A_[i].dot(deltas[i]).scalMul(learningRate_).print();

    
    std::cout << "Got this 2" << std::endl;
    
    // A_[i] = A_[i].transpose(); 
    // weights_[i].add(A_[i].dot(deltas[i])).scalMul(learningRate_);
    Matrix<T> dW = dot(A_[i].transpose(), deltas[i]);
    std::cout << "Got this 3" << std::endl;
    // dW = dW.scalMul(learningRate_);
    std::cout << "Got this 4" << std::endl;
    weights_[i].print();
    dW.print();
    weights_[i].add(dW);
    std::cout << "Got this 5" << std::endl;
    biases_[i].add(deltas[i].sum(1).scalMul(learningRate_));
    std::cout << "Got this 6" << std::endl;
    std::cout << "\n" << std::endl;
    
    // biases_[i] = add(biases_[i], deltas[i].sum(1).scalMul(learningRate_)); 
  }


}

template <typename S, typename T>
void NeuralNetwork<S, T>::train(Matrix<T> X, Matrix<T> y, size_t batch_size, float learning_rate, int epochs, bool verbose) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::cout << "Starting forward pass." << std::endl; 
    Matrix<T> output = forward(X);
    // std::cout << "Forward pass done." << std::endl; 
    std::cout << "Starting backward pass." << std::endl; 
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