
#pragma once

#include <functional>

#include "./Matrix.h"
#include "./Activation.h"

// Simple feed forward neural network.
template <typename S, typename T>
class NeuralNetwork {

private:

  // ____________________________________________________________________________
  // NeuralNetwork settings:
  // ____________________________________________________________________________

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
  std::vector<std::function<Matrix<T>(Matrix<T>&)>> activationFunctions_;

  // Activation function derivatives.
  std::vector<std::function<Matrix<T>(Matrix<T>&)>> activationFunctionDerivatives_;

// ____________________________________________________________________________
// Forward, backward propagation:
// ____________________________________________________________________________

  // Storing activations.
  std::vector<Matrix<T>> A_;

  // Stroing Zs
  std::vector<Matrix<T>> Z_;


  // Forward propagation. 
  Matrix<T> forward(Matrix<T> X);

  // Backpropagation.
  void backward(Matrix<T> X, Matrix<T> y);

public:
  // ____________________________________________________________________________
  // Constructor:
  // ____________________________________________________________________________
  
  NeuralNetwork(std::vector<size_t> layers, std::vector<Activation> activation_functions, float learning_rate=0.1f, InitState state = InitState::RANDOM);

  // ____________________________________________________________________________
  // Training and evaluation:
  // ____________________________________________________________________________
  
  // Trains the neural net.
  void train(Matrix<T> X, Matrix<T> y, size_t batchSize=1, float learningRate=0.1f, int epochs=1, bool verbose=false);
  void train(std::vector<std::vector<T>> X, std::vector<std::vector<T>> y, size_t batchSize, float learningRate);

  // Evaluates neural net.
  void evaluate(Matrix<T> X, Matrix<T> y);
  void evaluate(std::vector<std::vector<T>> X, std::vector<std::vector<T>> y);


  Matrix<T> act(Matrix<T> X);

};