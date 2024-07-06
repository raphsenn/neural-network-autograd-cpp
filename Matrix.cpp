
#include <cstdlib>

#include "./Matrix.h"
#include "./Utils.h"


template <typename T>
Matrix<T>::Matrix(std::size_t rows, std::size_t cols, InitState state) : rows_(rows), cols_(cols) {
  // Allocate memory.
  matrix_ = new T*[rows_];
  for (std::size_t row = 0; row < rows_; ++row) {
    matrix_[row] = new T[cols_];
  }
  // Handle InitState for matrix entrys.
  switch (state) {
    case InitState::ZERO: fillZeros();
    case InitState::RANDOM: fillRandom(); 
  }
}

template <typename T> Matrix<T>::~Matrix() {
  // Deallocate memory. 
  for (std::size_t row = 0; row < rows_; ++row) {
    delete[] matrix_[row];
  }
  delete[] matrix_;
}


template <typename T> std::size_t Matrix<T>::getRows() { return rows_; }

template <typename T> std::size_t Matrix<T>::getCols() { return cols_; }

template <typename T> void Matrix<T>::fillZeros() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = zero_value<T>::value();
    }
  }
}

template <typename T> void Matrix<T>::fillRandom() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = rand() % 1000;
    }
  }
}

// Explicit instantiations for float.
template class Matrix<float>;