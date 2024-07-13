
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
Matrix<T> relu(Matrix<T>& X) { X.maximum(value<T>::zero()); return X; }

template <typename T>
Matrix<T> relu_derivative(Matrix<T>& X) {
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      X[row][col] = (X[row][col] > value<T>::zero()) ? value<T>::one() : value<T>::zero();
    }
  }
  return X;
} 

// ____________________________________________________________________________
// Step
template <typename T>
Matrix<T> step(Matrix<T>& X) { 
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      X[row][col] = (X[row][col] >= value<T>::zero()) ? value<T>::one() : value<T>::zero();
    }
  }
  return X;
 }

template <typename T>
Matrix<T> step_derivative(Matrix<T>& X) {
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      X[row][col] = value<T>::zero();
    }
  }
  return X;
}


// ____________________________________________________________________________
// Explicit instantiations for float.
template Matrix<float> linear<float>(Matrix<float>& X);
template Matrix<float> linear_derivative<float>(Matrix<float>& X);
template Matrix<float> relu<float>(Matrix<float>& X);
template Matrix<float> relu_derivative<float>(Matrix<float>& X);
template Matrix<float> step<float>(Matrix<float>& X);
template Matrix<float> step_derivative<float>(Matrix<float>& X);