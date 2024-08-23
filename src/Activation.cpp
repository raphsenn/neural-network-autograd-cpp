
#include "./Activation.h"
#include "./Utils.h"

// ____________________________________________________________________________
// Linear
template <typename T> Matrix<T> linear(const Matrix<T> &X) { return X; }

// ____________________________________________________________________________
// Linear derivative
template <typename T> Matrix<T> linear_derivative(const Matrix<T> &X) {
  return Matrix<T>(X.getRows(), X.getCols(), InitState::ONES);
}

// ____________________________________________________________________________
// Relu
template <typename T> Matrix<T> relu(const Matrix<T> &X) {
  Matrix<T> result = X;
  result.maximum(value<T>::zero());
  return result;
}

// ____________________________________________________________________________
// Relu derivative
template <typename T> Matrix<T> relu_derivative(const Matrix<T> &X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] =
          (X[row][col] > value<T>::zero()) ? value<T>::one() : value<T>::zero();
    }
  }
  return result;
}

// ____________________________________________________________________________
// Step
template <typename T> Matrix<T> step(const Matrix<T> &X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = (X[row][col] >= value<T>::zero()) ? value<T>::one()
                                                           : value<T>::zero();
    }
  }
  return result;
}

// ____________________________________________________________________________
// Step derivative
template <typename T> Matrix<T> step_derivative(const Matrix<T> &X) {
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
template <typename T> Matrix<T> sigmoid(const Matrix<T> &X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = value<T>::one() /
                         (value<T>::one() + value<T>::e(-X.getValue(row, col)));
    }
  }
  return result;
}

// ____________________________________________________________________________
// Sigmoid derivative
template <typename T> Matrix<T> sigmoid_derivative(const Matrix<T> &X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] =
          (value<T>::one() /
           (value<T>::one() + value<T>::e(-X.getValue(row, col)))) *
          (value<T>::one() -
           (value<T>::one() /
            (value<T>::one() + value<T>::e(-X.getValue(row, col)))));
    }
  }
  return result;
}

// ____________________________________________________________________________
// Tanh
template <typename T> Matrix<T> tanh(const Matrix<T> &X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = value<T>::tanh(X.getValue(row, col));
    }
  }
  return result;
}

// ____________________________________________________________________________
// Tanh derivative
template <typename T> Matrix<T> tanh_derivative(const Matrix<T> &X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] =
          value<T>::one() - std::pow(value<T>::tanh(X.getValue(row, col)), 2.0);
    }
  }
  return result;
}

// ____________________________________________________________________________
// Softmax
template <typename T> Matrix<T> softmax(const Matrix<T> &X) {
  size_t rows = X.getRows();
  size_t cols = X.getCols();
  Matrix<T> result(rows, cols, InitState::EMPTY);

  for (size_t col = 0; col < cols; ++col) {
    T sum_exp = 0;
    // Compute the sum of exponentials
    for (size_t row = 0; row < rows; ++row) {
      sum_exp += std::exp(X.getValue(row, col));
    }
    // Normalize by the sum of exponentials
    for (size_t row = 0; row < rows; ++row) {
      result[row][col] = std::exp(X.getValue(row, col)) / sum_exp;
    }
  }

  return result;
}

// ____________________________________________________________________________
// Softmax derivative
template <typename T> Matrix<T> softmax_derivative(const Matrix<T> &X) {
  Matrix<T> softmax_vals = softmax(X);
  size_t rows = X.getRows();
  size_t cols = X.getCols();
  Matrix<T> result(rows, cols, InitState::EMPTY);

  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      T s = softmax_vals.getValue(row, col);
      result[row][col] = s * (1 - s);
    }
  }

  return result;
}

// ____________________________________________________________________________
// Exp
template <typename T> Matrix<T> exp(const Matrix<T> &X) {
  Matrix<T> result(X.getRows(), X.getCols(), InitState::EMPTY);
  for (size_t row = 0; row < X.getRows(); ++row) {
    for (size_t col = 0; col < X.getCols(); ++col) {
      result[row][col] = value<T>::e(X.getValue(row, col));
    }
  }
  return result;
}

// ____________________________________________________________________________
// Explicit instantiations for float.
template Matrix<float> linear<float>(const Matrix<float> &X);
template Matrix<float> linear_derivative<float>(const Matrix<float> &X);

template Matrix<float> relu<float>(const Matrix<float> &X);
template Matrix<float> relu_derivative<float>(const Matrix<float> &X);

template Matrix<float> step<float>(const Matrix<float> &X);
template Matrix<float> step_derivative<float>(const Matrix<float> &X);

template Matrix<float> tanh<float>(const Matrix<float> &X);
template Matrix<float> tanh_derivative<float>(const Matrix<float> &X);

template Matrix<float> sigmoid<float>(const Matrix<float> &X);
template Matrix<float> sigmoid_derivative<float>(const Matrix<float> &X);

template Matrix<float> softmax<float>(const Matrix<float> &X);
template Matrix<float> softmax_derivative<float>(const Matrix<float> &X);

template Matrix<float> exp<float>(const Matrix<float> &X);