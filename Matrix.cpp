
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
  matrix_ = std::vector(rows_, std::vector<T>(cols_));
  // Handle InitState for matrix entrys.
  switch (state) {
    case InitState::ZERO: fillZeros(); break;
    case InitState::RANDOM: fillRandom(); break;
    case InitState::ONES: fillOnes(); break;
    case InitState::EMPTY: break;
  }
}

// ____________________________________________________________________________
template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& other) {
  if (rows_ != other.size() || cols_ != other[0].size()) {
    matrix_ = std::vector<std::vector<T>>(other.size(), std::vector<T>(other[0].size()));
    rows_ = other.size();
    cols_ = other[0].size();
  }
  // Copy elements from 2D vector to matrix_.
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = other[row][col];
    }
  }
}

// ____________________________________________________________________________
// Operators:
// ____________________________________________________________________________

// ____________________________________________________________________________
template <typename T>
Matrix<T>& Matrix<T>::operator=(const std::vector<std::vector<T>> other) {
  if (rows_ != other.size() || cols_ != other[0].size()) {
    rows_ = static_cast<size_t> (other.size());
    cols_ = static_cast<size_t> (other[0].size());
    matrix_ = std::vector<std::vector<T>>(other.size(), std::vector<T>(other[0].size())); 
  } 
  
  // Perform copy.
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = other[row][col];
    }
  }
  return *this;
}

// ____________________________________________________________________________
template <typename T>
std::vector<T>& Matrix<T>::operator[](const std::size_t row) {
  // Handle if row >= rows_. 
  if (row >= rows_) { throw std::out_of_range("Row index out of range"); }
  return matrix_[row];
};

// ____________________________________________________________________________
template <typename T>
const std::vector<T>& Matrix<T>::operator[](const std::size_t col) const {
  // Handle if col >= cols_. 
  if (col >= cols_) { throw std::out_of_range("Col index out of range"); }
  return matrix_[col];
};

// ____________________________________________________________________________
template <typename T>
bool Matrix<T>::operator==(const Matrix<T> other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) { return false; }
  
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      if (matrix_[row][col] != other.matrix_[row][col]) { return false; }
    }
  }
  return true;
}

// ____________________________________________________________________________
// Linear Algebra operations:
// ____________________________________________________________________________

// ____________________________________________________________________________
template <typename T>
Matrix<T> Matrix<T>::add(const Matrix<T>& other) {
  // Scalar addition. 
  if (other.rows_ == 1 && other.cols_ == 1) {
  // Perform matrix addition with scalar value.
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = matrix_[row][col] + other.matrix_[0][0];
    }
  }
  return *this;
  }

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
  return *this;
}

// ____________________________________________________________________________
template <typename T>
Matrix<T> Matrix<T>::sub(const Matrix<T>& other) {
  // Check if matrices are in the same vectorspace.
  if (rows_ != other.rows_ || cols_ != other.cols_) { 
    throw std::invalid_argument("Matrices dimensions do not match for subtraction.");
  }
  // Perform matrix addition.
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = matrix_[row][col] - other.matrix_[row][col];
    }
  }
  return *this;
}

// ____________________________________________________________________________
template <typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& other) {
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
  return *this;
}

// ____________________________________________________________________________
template <typename T>
Matrix<T> Matrix<T>::dotElementWise(const Matrix<T>& other) {
  // Check if matrices are in the same vectorspace.
  if (rows_ != other.rows_ || cols_ != other.cols_) { 
    throw std::invalid_argument("Matrices dimensions do not match for elementt wise multiplication.");
  }
  
  // Perform element wise multiplication.
  Matrix<T> C(rows_, other.cols_, InitState::ZERO); 
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < other.cols_; ++col) { 
        C[row][col] += matrix_[row][col] * other.matrix_[row][col];
    }
  }
  *this = std::move(C);
  return *this;
}

// ____________________________________________________________________________
template <typename T>
Matrix<T> Matrix<T>::transpose() {
  Matrix<T> transposed(cols_, rows_, InitState::EMPTY);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      transposed[col][row] = matrix_[row][col]; 
    }
  }
  *this = std::move(transposed);
  return *this;
}

// ____________________________________________________________________________
template <typename T>
void Matrix<T>::maximum(T inf) {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      if (matrix_[row][col] < inf) {matrix_[row][col] = inf; }
    }
  }
}

// ____________________________________________________________________________
template <typename T>
Matrix<T> Matrix<T>::sum() const {
  Matrix<T> sum(1, 1, InitState::ZERO);
  // T sum = value<T>::zero();
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      sum.matrix_[0][0] += matrix_[row][col];
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
      matrix_[row][col] = value<T>::zero();
    }
  }
}

// ____________________________________________________________________________
template <typename T>
void Matrix<T>::fillRandom() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = value<T>::random();
    }
  }
}

// ____________________________________________________________________________
template <typename T>
void Matrix<T>::fillOnes() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = value<T>::one();
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