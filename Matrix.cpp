
#include <cstdlib>
#include <cstddef>
#include <stdexcept>

#include "./Matrix.h"
#include "./Utils.h"


template <typename T>
Matrix<T>::Matrix(std::size_t rows, std::size_t cols, InitState state) : rows_(rows), cols_(cols) {
  if (rows_ <= 0 || cols_ <= 0) { throw std::invalid_argument("Rows or cols must be > 0");}
  // Allocate memory.
  matrix_ = new T*[rows_];
  for (std::size_t row = 0; row < rows_; ++row) {
    matrix_[row] = new T[cols_];
  }
  // Handle InitState for matrix entrys.
  switch (state) {
    case InitState::ZERO: fillZeros(); break;
    case InitState::RANDOM: fillRandom(); break;
  }
}

template <typename T>
Matrix<T>::~Matrix() {
  // Deallocate memory. 
  for (std::size_t row = 0; row < rows_; ++row) {
    delete[] matrix_[row];
  }
  delete[] matrix_;
}
template <typename T>
T* Matrix<T>::operator[](std::size_t row) {
  // Handle if row >= rows_. 
  if (row >= rows_) { throw std::out_of_range("Row index out of range"); }
  return matrix_[row];
};

template <typename T>
const T* Matrix<T>::operator[](std::size_t col) const {
  // Handle if col >= cols_. 
  if (col >= cols_) { throw std::out_of_range("Row index out of range"); }
  return matrix_[col];
};


template <typename T>
std::size_t Matrix<T>::getRows() const { return rows_; }

template <typename T>
std::size_t Matrix<T>::getCols() const { return cols_; }

template <typename T>
void Matrix<T>::fillZeros() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = zero_value<T>::value();
    }
  }
}

template <typename T>
void Matrix<T>::fillRandom() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = rand() % 1000;
    }
  }
}

template <typename T>
void Matrix<T>::print() {
  std::cout << "matrix([";
  for (size_t row = 0; row < rows_; ++row) {
    std::cout << "[";
    for (size_t col = 0; col < cols_; ++col) {
      std::cout << matrix_[row][col];
      if (col < cols_ - 1) std::cout << ", ";
    }
    std::cout << "]";
    if (row < rows_ - 1) std::cout << ",\n";
  }
  std::cout << "])\n";
}


// Explicit instantiations for float and double.
template class Matrix<float>;
template class Matrix<double>;