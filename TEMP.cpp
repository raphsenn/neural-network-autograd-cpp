

#include "./Matrix.h"

// ____________________________________________________________________________
template <typename T>
Matrix<T> &Matrix<T>::operator=(const std::vector<std::vector<T>> other) {
  // Perform copy.
  rows_ = static_cast<size_t> (other.size());
  cols_ = static_cast<size_t> (other[0].size());
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = other[row][col];
    }
  }
  return *this;
}

// ____________________________________________________________________________
template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& matrix) {
  rows_ = static_cast<size_t>( matrix.size());
  cols_ = static_cast<size_t>( matrix[0].size());
  
  // Copy elements from 2D vector to matrix_.
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = matrix[row][col];
    }
  }
}