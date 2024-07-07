
#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "./Matrix.h"
#include "./Utils.h"

// ____________________________________________________________________________
// Constructors:
// ____________________________________________________________________________

// ____________________________________________________________________________
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
    case InitState::EMPTY: break;
  }
}

// ____________________________________________________________________________
template <typename T>
Matrix<T>::Matrix(const Matrix<T>& matrix) {
  rows_ = matrix.rows_; 
  cols_ = matrix.cols_; 

  // Allocate new memory.
  matrix_ = new T*[rows_];
  for (std::size_t row = 0; row < rows_; ++row) {
    matrix_[row] = new T[cols_];
  }

  // Copy elements.
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = matrix.matrix_[row][col];
    }
  }
}

// ____________________________________________________________________________
template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& matrix) {
  rows_ = static_cast<size_t>( matrix.size());
  cols_ = static_cast<size_t>( matrix[0].size());
  
  // Allocate memory.
  matrix_ = new T*[rows_];
  for (std::size_t row = 0; row < rows_; ++row) {
    matrix_[row] = new T[cols_];
  }
  
  // Copy elements from 2D vector to matrix_.
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = matrix[row][col];
    }
  }
}

// ____________________________________________________________________________
template <typename T>
Matrix<T>::Matrix(Matrix<T>&& matrix) {
  rows_ = matrix.rows_; 
  cols_ = matrix.cols_; 
  matrix_ = matrix.matrix_;
  matrix.rows_ = 0;
  matrix.cols_ = 0;
}

// ____________________________________________________________________________
template <typename T>
Matrix<T>::~Matrix() {
  // Deallocate memory. 
  for (std::size_t row = 0; row < rows_; ++row) {
    delete[] matrix_[row];
  }
  delete[] matrix_;
}

// ____________________________________________________________________________
// Operators:
// ____________________________________________________________________________

// ____________________________________________________________________________
template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix& other) {
  // Handle self assignment. 
  if (this == &other) { return *this; }
  
  // Deallocate memory. 
  for (std::size_t row = 0; row < rows_; ++row) {
    delete[] matrix_[row];
  }
  delete[] matrix_;

  // Perform copy.
  rows_ = other.rows_;
  cols_ = other.cols_;
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = other.matrix_[row][col];
    }
  }
  return *this;
}

// ____________________________________________________________________________
template <typename T>
Matrix<T> &Matrix<T>::operator=(const std::vector<std::vector<T>> other) {
  // Deallocate memory. 
  for (std::size_t row = 0; row < rows_; ++row) {
    delete[] matrix_[row];
  }
  delete[] matrix_;

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
Matrix<T> &Matrix<T>::operator=(Matrix&& other) {
  // Deallocate memory. 
  for (std::size_t row = 0; row < rows_; ++row) {
    delete[] matrix_[row];
  }
  delete[] matrix_;
  rows_ = other.rows_;
  cols_ = other.cols_;
  matrix_ = other.matrix_;
  other.rows_ = 0;
  other.cols_ = 0;
  other.matrix_ = nullptr;
  return *this;
}

// ____________________________________________________________________________
template <typename T>
T* Matrix<T>::operator[](const std::size_t row) {
  // Handle if row >= rows_. 
  if (row >= rows_) { throw std::out_of_range("Row index out of range"); }
  return matrix_[row];
};

// ____________________________________________________________________________
template <typename T>
const T* Matrix<T>::operator[](const std::size_t col) const {
  // Handle if col >= cols_. 
  if (col >= cols_) { throw std::out_of_range("Row index out of range"); }
  return matrix_[col];
};

// ____________________________________________________________________________
// Linear Algebra operations:
// ____________________________________________________________________________

// ____________________________________________________________________________
template <typename T>
void Matrix<T>::add(const Matrix<T>& other) {
  // Check if matrices are in the same vectorspace.
  if (rows_ != other.rows_ || cols_ != other.cols_) { 
    throw std::invalid_argument("Matrices dimensions do not match for addition.");
  }
  // Perform matrix addition.
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = matrix_[row][col] + other.matrix_[row][col];
    }
  }
}

// ____________________________________________________________________________
template <typename T>
void Matrix<T>::dot(const Matrix<T>& other) {
  // Check if matrices are in the same vectorspace.
  if (cols_ != other.rows_) { 
    throw std::invalid_argument("Matrices dimensions do not match for multiplication.");
  }
  
  // Perform matrix multiplication.
  // Really expensive, maybie work on this later.
  Matrix<T> C(rows_, other.cols_, InitState::ZERO); 
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t k = 0; k < other.cols_; ++k) { 
      for (size_t j = 0; j < cols_; ++j) {
        C[i][k] += matrix_[i][j] * other.matrix_[j][k];
      }
    }
  }
  *this = std::move(C);
}

// ____________________________________________________________________________
template <typename T>
void Matrix<T>::transpose() {
  Matrix<T> transposed(cols_, rows_, InitState::EMPTY);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      transposed[col][row] = matrix_[row][col]; 
    }
  }
  *this = std::move(transposed);
}

// ____________________________________________________________________________
template <typename T>
T Matrix<T>::sum() const {
  T sum = zero_value<T>::value();
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      sum = sum + matrix_[row][col];
    }
  }
  return sum;
}

// ____________________________________________________________________________
template <typename T>
std::size_t Matrix<T>::getRows() const { return rows_; }

// ____________________________________________________________________________
template <typename T>
std::size_t Matrix<T>::getCols() const { return cols_; }

// ____________________________________________________________________________
template <typename T>
void Matrix<T>::fillZeros() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = zero_value<T>::value();
    }
  }
}

// ____________________________________________________________________________
template <typename T>
void Matrix<T>::fillRandom() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = random_value<T>::value();
    }
  }
}

// ____________________________________________________________________________
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

// ____________________________________________________________________________
// Explicit instantiations for int, float and double.
template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;