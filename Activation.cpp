
#include "./Activation.h"
#include "./Utils.h"


// ____________________________________________________________________________
// Linear
template <typename T>
Matrix<T> linear(Matrix<T>& X) { return X; }

template <typename T>
Matrix<T> linear_derivative(Matrix<T>& X) { return Matrix<T>(X.getRows(), X.getCols(), InitState::ONES); }

// ____________________________________________________________________________
// Relu
template <typename T>
Matrix<T> relu(Matrix<T>& X) { 
  Matrix<T> result = X;
  result.maximum(value<T>::zero());
  return result;
}

template <typename T>
Matrix<T> relu_derivative(Matrix<T>& X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = (X[row][col] > value<T>::zero()) ? value<T>::one() : value<T>::zero();
    }
  }
  return result;
} 

// ____________________________________________________________________________
// Step
template <typename T>
Matrix<T> step(Matrix<T>& X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = (X[row][col] >= value<T>::zero()) ? value<T>::one() : value<T>::zero();
    }
  }
  return result;
 }

template <typename T>
Matrix<T> step_derivative(Matrix<T>& X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = value<T>::zero();
    }
  }
  return result;
}

// ____________________________________________________________________________
// Sigmoid
template <typename T>
Matrix<T> sigmoid(Matrix<T>& X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = value<T>::one() / (value<T>::one() + value<T>::e(-X[row][col]));
    }
  }
  return result;
 }

template <typename T>
Matrix<T> sigmoid_derivative(Matrix<T>& X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = (value<T>::one() / (value<T>::one() + value<T>::e(-X[row][col]))) * (value<T>::one() - (value<T>::one() / (value<T>::one() + value<T>::e(-X[row][col]))));
    }
  }
  return result;
}


// ____________________________________________________________________________
// Explicit instantiations for float.
template Matrix<float> linear<float>(Matrix<float>& X);
template Matrix<float> linear_derivative<float>(Matrix<float>& X);
template Matrix<float> relu<float>(Matrix<float>& X);
template Matrix<float> relu_derivative<float>(Matrix<float>& X);
template Matrix<float> step<float>(Matrix<float>& X);
template Matrix<float> step_derivative<float>(Matrix<float>& X);

template Matrix<float> sigmoid<float>(Matrix<float>& X);
template Matrix<float> sigmoid_derivative<float>(Matrix<float>& X);