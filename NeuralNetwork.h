
#pragma once

#include <functional>

#include "./Activation.h"
#include "./Matrix.h"

// Simple feed forward neural network.
template <typename T> class NeuralNetwork {

private:
  // ____________________________________________________________________________
  // NeuralNetwork settings:

  // Learning rate.
  float learningRate_;

  // Layer sizes.
  std::vector<size_t> layerSizes_;

  // Number of layers.
  size_t numLayers_;

  // Weights.
  std::vector<Matrix<T>> weights_;

  // Biases.
  std::vector<Matrix<T>> biases_;

  // Activation functions.
  std::vector<std::function<Matrix<T>(Matrix<T> &)>> activationFunctions_;

  // Activation function derivatives.
  std::vector<std::function<Matrix<T>(Matrix<T> &)>>
      activationFunctionDerivatives_;

  // ____________________________________________________________________________
  // Forward, backward propagation:

  // Storing activations.
  std::vector<Matrix<T>> A_;

  // Stroing Zs
  std::vector<Matrix<T>> Z_;

  // Forward propagation.
  Matrix<T> forward(const Matrix<T> &X);

  // Backpropagation.
  void backward(Matrix<T> y);

public:
  // ____________________________________________________________________________
  // Constructor:

  NeuralNetwork() = default;

  NeuralNetwork(std::vector<size_t> layers,
                std::vector<Activation> activation_functions,
                float learning_rate = 0.1f,
                InitState state = InitState::RANDOM);

  // ____________________________________________________________________________
  // Training and evaluation:

  // Trains the neural net.
  void train(Matrix<T> X, Matrix<T> y, size_t batchSize = 1,
             float learningRate = 0.1f, int epochs = 1, bool verbose = false);

  // Generates an output with input data X.
  Matrix<T> act(const Matrix<T> &X);

  // Calculate loss (Mean Squared Error).
  float loss(Matrix<T>& out, Matrix<T>& y);

  // Calculates accuracy.
  float getAccuracy(Matrix<T>& out, Matrix<T>& y, float threshold=0.3f);

  // Evaluates neural net.
  // Calculates performance metrics precision, recall and accuracy.
  // Prints performance metrics.
  void evaluate(Matrix<T>& X, Matrix<T>& y, bool binary=false);

  // Saves weights and biases to binary file.
  void save(std::string fileName="neural_network_data.bin");

  // Loads weights and biases from a binary file.
  void load(std::string fileName);
};