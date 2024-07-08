
#pragma once

#include "./Matrix.h"
#include "./Activation.h"

// Simple feed forward neural network.
template <typename T>
class NeuralNetwork {

private:
  // Learning rate. 
  float learningRate_;

  // Layer sizes.
  std::vector<size_t> layerSizes_;

  // Number of layers.
  int numLayers_;

  // Weights.
  std::vector<Matrix<T>> weights_; 
  
  // Biases.
  std::vector<Matrix<T>> biases_; 

  // Activation functions.
  std::vector<Matrix<T>> activationFunctions_;

  // Activation function derivatives.
  std::vector<Matrix<T>> activationFunctionDerivatives_;

  // Forward propagation. 
  void forward();

  // Backpropagation.
  void backward();

public:
  // Constructor.
  NeuralNetwork(std::vector<size_t> layers, std::vector<Activation> activation_functions, float learning_rate=0.1f, InitState state = InitState::RANDOM);

  // Trains the neural net.
  void train(Matrix<T> X, Matrix<T> y, size_t batchSize, float learningRate);
  void train(std::vector<std::vector<T>> X, std::vector<std::vector<T>> y, size_t batchSize, float learningRate);

  // Evaluates neural net.
  void evaluate(Matrix<T> X, Matrix<T> y);
  void evaluate(std::vector<std::vector<T>> X, std::vector<std::vector<T>> y);
};