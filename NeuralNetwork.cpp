
#include <algorithm>
#include <fstream>

#include "./NeuralNetwork.h"

// ____________________________________________________________________________
template <typename T>
NeuralNetwork<T>::NeuralNetwork(std::vector<size_t> layers,
                                std::vector<Activation> activation_functions,
                                float learning_rate, InitState state) {
  layerSizes_ = layers;
  numLayers_ = layers.size();
  learningRate_ = learning_rate;

  // Create weights and biases.
  for (size_t i = 0; i < layers.size() - 1; ++i) {
    weights_.push_back(Matrix<T>(layers[i], layers[i + 1], state));
    biases_.push_back(Matrix<T>(1, layers[i + 1], state));
  }

  for (const auto &act : activation_functions) {
    // Not pretty i guess, maybie better to store pointers.
    // https://en.cppreference.com/w/cpp/utility/functional/function
    switch (act) {
    case Activation::linear:
      activationFunctions_.push_back([](Matrix<T> &X) { return linear(X); });
      activationFunctionDerivatives_.push_back(
          [](Matrix<T> &X) { return linear_derivative(X); });
      break;
    case Activation::relu:
      activationFunctions_.push_back([](Matrix<T> &X) { return relu(X); });
      activationFunctionDerivatives_.push_back(
          [](Matrix<T> &X) { return relu_derivative(X); });
      break;
    case Activation::step:
      activationFunctions_.push_back([](Matrix<T> &X) { return step(X); });
      activationFunctionDerivatives_.push_back(
          [](Matrix<T> &X) { return step_derivative(X); });
      break;
    case Activation::sigmoid:
      activationFunctions_.push_back([](Matrix<T> &X) { return sigmoid(X); });
      activationFunctionDerivatives_.push_back(
          [](Matrix<T> &X) { return sigmoid_derivative(X); });
      break;
    case Activation::softmax:
      break;
    case Activation::tanh:
      break;
    case Activation::maxout:
      break;
    }
  }
}

// ____________________________________________________________________________
// Forward propagation:
template <typename T> Matrix<T> NeuralNetwork<T>::forward(const Matrix<T> &X) {

  // Pre-active values (weighted sums).
  Z_.clear();

  // Activations (of weighted sums).
  A_.clear();

  // Initialize activations with input data X.
  A_.push_back(X);

  // Forward propagation.
  // In a nutshell:
  //
  // Weightes sums Z:
  // Z_[0] = dot(X, W[0]) + BIAS[0]
  // Z_[1] = dot(ACT_0(dot(X, W[0]) + BIAS[0]), W[1]) + BIAS[1]
  // Z_[2] = dot(ACT_1(dot(ACT_0(dot(X, W[0]) + BIAS[0]), W[1]) + BIAS[1]),
  // W[2]) + BIAS[2]
  // ...
  //
  // Activations A:
  // A_[0] = X
  // A_[1] = ACT_0(dot(X, W[0]) + BIAS[0])
  // A_[2] = ACT_1(dot(ACT_0(dot(X, W[0]) + BIAS[0]), W[1]) + BIAS[1])
  // ...
  // A_[n] = ACT_n(dot(... (ACT_1(dot(ACT_0(dot(X, W[0]) + BIAS[0]), W[1]) +
  // BIAS[1]) ...W[n]) + BIAS[n])

  // Loop through each layer to perform forward propagation.
  for (size_t i = 0; i < numLayers_ - 1; ++i) {
    // Weighted sums:
    // Z_[i] = dot(A[i], W[i]) + BIAS[i]
    Matrix<T> z = std::move(dot(A_.back(), weights_[i]));
    z = std::move(add(z, biases_[i]));
    Z_.push_back(z);

    // Activation of weighted sums:
    // A_[i + 1] = activate(dot(A[i], W[i]) + BIAS) = activate(Z_[i])
    Matrix<T> a = activationFunctions_[i](z);
    A_.push_back(a);
  }

  // Return final output of the network.
  return A_.back();
}

// ____________________________________________________________________________
// Backpropagation:
template <typename T> void NeuralNetwork<T>::backward(Matrix<T> y) {

  // __________________________________________________________________________
  // Backpropagation in a nutshell.
  //
  // 1. Calculate output error:
  // output error = output - y
  //
  // 2. Calculate delta for each layer by propagating the error backwards
  // through the network.
  //
  // 3. Calculate gradient of weights and biases.
  //
  // 4. Update weights and biases.
  // __________________________________________________________________________

  // Calculate output error.
  // Calculates: output - labels = output_error
  Matrix<T> output_error = std::move(sub(y, A_.back()));

  // Calculate the derivative of the activation function at the output layer.
  Matrix<T> activation_derivative =
      activationFunctionDerivatives_.back()(A_.back());

  // Compute delta for the output layer using element-wise multiplication of
  // error and activation derivative.
  Matrix<T> delta =
      std::move(dotElementWise(output_error, activation_derivative));

  // Store delta values in a vector for each layer.
  std::vector<Matrix<T>> deltas;
  deltas.push_back(delta);

  // Propagate the error backwards through the network.
  // This was kind of hard xd.
  for (size_t i = numLayers_ - 2; i > 0; --i) {
    // Transpose the weight matrix of the next layer
    Matrix<T> W = std::move(weights_[i].transpose_copy());

    // Calculate delta for the current layer
    // delta = (delta_next * W_next) * activation_derivative
    delta = dot(deltas.back(), W);
    activation_derivative = activationFunctionDerivatives_[i](A_[i]);
    delta = dotElementWise(delta, activation_derivative);
    deltas.push_back(delta);
  }
  std::reverse(deltas.begin(), deltas.end());

  // Update weights and biases.
  for (size_t i = 0; i < numLayers_ - 1; ++i) {
    // Compute weight gradients
    // dW = A_[i].transpose() * delta[i]
    Matrix<T> A_i_transposed = std::move(A_[i].transpose_copy());
    Matrix<T> dW = std::move(dot(A_i_transposed, deltas[i]));
    // Update weights.
    dW = dW.scalMul(learningRate_);
    weights_[i].add(dW);

    // Compute bias gradients
    // dB = sum(delta[i]) * learningRate.
    Matrix<T> dB = deltas[i].sum(1).scalMul(learningRate_);
    // Update Biases
    biases_[i] = add(biases_[i], dB);
  }
}

// ____________________________________________________________________________
template <typename T>
void NeuralNetwork<T>::train(Matrix<T> X, Matrix<T> y, size_t batch_size,
                             float learning_rate, int epochs, bool verbose) {
  if (learning_rate != 0.1f) {
    learningRate_ = learning_rate;
  }
  if (verbose) {
    std::cout << "Start training NeuralNetwork with parameters: " << std::endl;
    std::cout << "LearningRate: " << learningRate_ << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
  }
  // Start training the NeuralNetwork.
  for (int epoch = 0; epoch < epochs; ++epoch) {
    Matrix<T> output = forward(X);
    backward(y);
    if (verbose) {
      if (epoch % 100 == 0) {
        std::cout << "Epoch: " << epoch << std::endl;
      }
    }
  }
}

// ____________________________________________________________________________
template <typename T> Matrix<T> NeuralNetwork<T>::act(const Matrix<T> &X) {
  return forward(X);
}

// ____________________________________________________________________________
template <typename T>
void NeuralNetwork<T>::evaluate(Matrix<T> X, Matrix<T> y, bool binary) {
  Matrix<T> y_out = forward(X);
  float acc = 0.0f;
  for (size_t row = 0; row < y.getRows(); ++row) {
    for (size_t col = 0; col < y.getCols(); ++col) {
      if (y_out[row][col] == y[row][col]) {
        ++acc;
      };
    }
  }
  float accuracy = acc / static_cast<float>(y_out.getRows() * y_out.getCols());
  std::cout << "Accuracy: " << accuracy << "%" << std::endl;
}

// ____________________________________________________________________________
template <typename T> void NeuralNetwork<T>::save(std::string fileName) {
  std::ofstream outFile(fileName, std::ios::binary);

  // Write number of layers.
  outFile.write(reinterpret_cast<const char *>(&numLayers_),
                sizeof(numLayers_));

  // Write the sizes of each layer.
  for (const auto &size : layerSizes_) {
    outFile.write(reinterpret_cast<const char *>(&size), sizeof(size));
  }
  // Write weights
  for (const auto &weightMatrix : weights_) {
    auto data = weightMatrix.getData();
    std::size_t rows = weightMatrix.getRows();
    std::size_t cols = weightMatrix.getCols();
    outFile.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    for (const auto &row : data) {
      outFile.write(reinterpret_cast<const char *>(row.data()),
                    row.size() * sizeof(T));
    }
  }

  // Write biases
  for (const auto &biasMatrix : biases_) {
    auto data = biasMatrix.getData();
    std::size_t rows = biasMatrix.getRows();
    std::size_t cols = biasMatrix.getCols();
    outFile.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    for (const auto &row : data) {
      outFile.write(reinterpret_cast<const char *>(row.data()),
                    row.size() * sizeof(T));
    }
  }
  outFile.close();
}

// ____________________________________________________________________________
template <typename T> void NeuralNetwork<T>::load(std::string fileName) {
  std::ifstream inFile(fileName, std::ios::binary);

  if (!inFile) {
    throw std::runtime_error("Cannot open file for reading");
  }

  // Read the number of layers
  inFile.read(reinterpret_cast<char *>(&numLayers_), sizeof(numLayers_));

  // Read the sizes of each layer
  layerSizes_.resize(numLayers_);
  for (auto &size : layerSizes_) {
    inFile.read(reinterpret_cast<char *>(&size), sizeof(size));
  }

  // Read weights
  weights_.clear();
  for (size_t i = 0; i < numLayers_ - 1;
       ++i) { // Assuming weights are between layers
    std::size_t rows, cols;
    inFile.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    Matrix<T> weightMatrix(rows, cols, InitState::EMPTY);
    auto data = weightMatrix.getData();
    for (auto &row : data) {
      row.resize(cols);
      inFile.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(T));
    }
    weightMatrix = data;
    // weightMatrix.setData(data);
    weights_.push_back(weightMatrix);
  }

  // Read biases
  biases_.clear();
  for (size_t i = 0; i < numLayers_ - 1;
       ++i) { // Assuming biases are for each layer except the input
    std::size_t rows, cols;
    inFile.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    Matrix<T> biasMatrix(rows, cols, InitState::EMPTY);
    auto data = biasMatrix.getData();
    for (auto &row : data) {
      row.resize(cols);
      inFile.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(T));
    }
    biasMatrix = data;
    // biasMatrix.setData(data);
    biases_.push_back(biasMatrix);
  }

  inFile.close();
}

// ____________________________________________________________________________
// Implicit instanziation for float.
template class NeuralNetwork<float>;