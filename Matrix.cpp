
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "./Matrix.h"
#include "./Utils.h"

// ____________________________________________________________________________
// Constructors:
// ____________________________________________________________________________

// ____________________________________________________________________________
template <typename T>
Matrix<T>::Matrix(std::size_t rows, std::size_t cols, InitState state)
    : rows_(rows), cols_(cols) {
  if (rows_ <= 0 || cols_ <= 0) {
    throw std::invalid_argument("Rows or cols must be > 0");
  }
  matrix_ = std::vector(rows_, std::vector<T>(cols_));
  // Handle InitState for matrix entrys.
  switch (state) {
  case InitState::ZERO:
    fillZeros();
    break;
  case InitState::RANDOM:
    fillRandom();
    break;
  case InitState::ONES:
    fillOnes();
    break;
  case InitState::EMPTY:
    break;
  }
}

// ____________________________________________________________________________
template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &other) {
  if (other.empty() || other[0].empty()) {
    throw std::invalid_argument("Input vector must not be empty");
  }
  rows_ = other.size();
  cols_ = other[0].size();
  matrix_ = std::vector<std::vector<T>>(rows_, std::vector<T>(cols_));

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
Matrix<T> &Matrix<T>::operator=(const std::vector<std::vector<T>> other) {
  if (rows_ != other.size() || cols_ != other[0].size()) {
    rows_ = static_cast<size_t>(other.size());
    cols_ = static_cast<size_t>(other[0].size());
    matrix_ = std::vector<std::vector<T>>(other.size(),
                                          std::vector<T>(other[0].size()));
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
std::vector<T> &Matrix<T>::operator[](const std::size_t row) {
  // Handle if row >= rows_.
  if (row >= rows_) {
    throw std::out_of_range("Row index out of range");
  }
  return matrix_[row];
};

// ____________________________________________________________________________
template <typename T>
const std::vector<T> &Matrix<T>::operator[](const std::size_t col) const {
  // Handle if col >= cols_.
  if (col >= cols_) {
    throw std::out_of_range("Col index out of range");
  }
  return matrix_[col];
};

// ____________________________________________________________________________
template <typename T> bool Matrix<T>::operator==(const Matrix<T> other) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    return false;
  }

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      if (matrix_[row][col] != other.matrix_[row][col]) {
        return false;
      }
    }
  }
  return true;
}

// ____________________________________________________________________________
// Linear Algebra operations:
// ____________________________________________________________________________

// ____________________________________________________________________________
template <typename T> Matrix<T> &Matrix<T>::add(const Matrix<T> &other) {
  // Scalar addition.
  if (cols_ == other.cols_ && other.rows_ == 1) {
    // Perform matrix addition with scalar value.
    for (size_t row = 0; row < rows_; ++row) {
      for (size_t col = 0; col < cols_; ++col) {
        matrix_[row][col] = matrix_[row][col] + other.matrix_[0][col];
      }
    }
    return *this;
  }

  // Check if matrices are in the same vectorspace.
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument(
        "Matrices dimensions do not match for addition.");
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
template <typename T> Matrix<T> Matrix<T>::sub(const Matrix<T> &other) {
  // Check if matrices are in the same vectorspace.
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument(
        "Matrices dimensions do not match for subtraction.");
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
template <typename T> Matrix<T> Matrix<T>::dot(const Matrix<T> &other) {
  // Check if matrices are in the same vectorspace.
  if (cols_ != other.rows_) {
    throw std::invalid_argument(
        "Matrices dimensions do not match for multiplication.");
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
  cols_ = other.cols_;
  *this = std::move(C);
  return *this;
}

// ____________________________________________________________________________
template <typename T> Matrix<T> &Matrix<T>::transpose() {
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
template <typename T> Matrix<T> Matrix<T>::transpose_copy() {
  Matrix<T> transposed(cols_, rows_, InitState::EMPTY);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      transposed[col][row] = matrix_[row][col];
    }
  }
  return transposed;
}

// ____________________________________________________________________________
template <typename T> Matrix<T> Matrix<T>::scalMul(T scalar) {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] *= scalar;
    }
  }
  return *this;
}

// ____________________________________________________________________________
template <typename T> void Matrix<T>::maximum(T inf) {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      if (matrix_[row][col] < inf) {
        matrix_[row][col] = inf;
      }
    }
  }
}

// ____________________________________________________________________________
template <typename T> Matrix<T> Matrix<T>::sum(bool axis) const {
  if (!axis) {
    Matrix<T> sum(rows_, 1, InitState::ZERO);
    for (size_t row = 0; row < rows_; ++row) {
      for (size_t col = 0; col < cols_; ++col) {
        sum.matrix_[row][0] += matrix_[row][col];
      }
    }
    return sum;
  }
  Matrix<T> sum(1, cols_, InitState::ZERO);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      sum.matrix_[0][col] += matrix_[row][col];
    }
  }
  return sum;
}

// ____________________________________________________________________________
// More methods (public):
// ____________________________________________________________________________

// ____________________________________________________________________________
template <typename T> std::size_t Matrix<T>::getRows() const { return rows_; }

// ____________________________________________________________________________
template <typename T> std::size_t Matrix<T>::getCols() const { return cols_; }

// ____________________________________________________________________________
template <typename T> std::vector<std::vector<T>> Matrix<T>::getData() const {
  return matrix_;
}

// ____________________________________________________________________________
template <typename T> void Matrix<T>::fillZeros() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = value<T>::zero();
    }
  }
}

// ____________________________________________________________________________
template <typename T> void Matrix<T>::fillRandom() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = value<T>::random();
    }
  }
}

// ____________________________________________________________________________
template <typename T> void Matrix<T>::fillOnes() {
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      matrix_[row][col] = value<T>::one();
    }
  }
}

// ____________________________________________________________________________
template <typename T> void Matrix<T>::print() {
  std::cout << "matrix([";
  for (size_t row = 0; row < rows_; ++row) {
    std::cout << "[";
    for (size_t col = 0; col < cols_; ++col) {
      std::cout << matrix_[row][col];
      if (col < cols_ - 1)
        std::cout << ", ";
    }
    std::cout << "]";
    if (row < rows_ - 1)
      std::cout << ",\n";
  }
  std::cout << "])\n";
}

// ____________________________________________________________________________
template <typename T>
T Matrix<T>::getValue(const size_t row, const size_t col) const {
  if (row >= rows_) {
    throw std::out_of_range("Row index out of range.");
  }
  if (col >= cols_) {
    throw std::out_of_range("Col index out of range.");
  }
  return matrix_[row][col];
}

// ____________________________________________________________________________
// Explicit instantiations (for Matrix class) for int, float and double.
template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;

// ____________________________________________________________________________
// Linear Algebra functions:
// ____________________________________________________________________________

// ____________________________________________________________________________
template <typename T> Matrix<T> dot(Matrix<T> &A, Matrix<T> &B) {
  // Check if matrices are in the same vectorspace.
  if (A.getCols() != B.getRows()) {
    throw std::invalid_argument(
        "Matrices dimensions do not match for multiplication.");
  }

  // Perform matrix multiplication (really expensive).
  Matrix<T> C(A.getRows(), B.getCols(), InitState::ZERO);
  for (size_t i = 0; i < A.getRows(); ++i) {
    for (size_t j = 0; j < B.getCols(); ++j) {
      T sum = value<T>::zero();
      for (size_t k = 0; k < A.getCols(); ++k) {
        // C[i][k] += A[i][j] * B[j][k];
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
  return C;
}

// ____________________________________________________________________________
template <typename T> Matrix<T> add(Matrix<T> &A, Matrix<T> &B) {
  // Case 1: Scalar addition.
  // Case 1.1:
  // [[1, 2], [3, 4]] + [[10, 10]] = [[11, 12], [13, 14]]
  if (A.getCols() == B.getCols() && B.getRows() == 1) {
    Matrix<T> C(A.getRows(), A.getCols(), InitState::ZERO);
    for (size_t row = 0; row < A.getRows(); ++row) {
      for (size_t col = 0; col < A.getCols(); ++col) {
        C[row][col] = A[row][col] + B[0][col];
      }
    }
    return C;
  }
  // Case 1.2:
  // [[1, 2], [3, 4]] + [[10], [10]] = [[11, 12], [13, 14]]
  if (A.getRows() == B.getRows() && B.getCols() == 1) {
    Matrix<T> C(A.getRows(), A.getCols(), InitState::ZERO);
    for (size_t row = 0; row < A.getRows(); ++row) {
      for (size_t col = 0; col < A.getCols(); ++col) {
        C[row][col] = A[row][col] + B[row][0];
      }
    }
    return C;
  }
  // Case 2: Addition of Matrix in same vectorspace.
  // Check if matrices are in the same vectorspace.
  if (A.getRows() != B.getCols() || A.getCols() != B.getCols()) {
    throw std::invalid_argument(
        "Matrices dimensions do not match for addition.");
  }
  // Perform matrix addition.
  Matrix<T> C(A.getRows(), A.getCols(), InitState::ZERO);
  for (size_t row = 0; row < A.getRows(); ++row) {
    for (size_t col = 0; col < A.getCols(); ++col) {
      C[row][col] = A[row][col] + B[row][col];
    }
  }
  return C;
}

// ____________________________________________________________________________
template <typename T> Matrix<T> sub(Matrix<T> &A, Matrix<T> &B) {
  // Scalar addition.
  if (A.getCols() == B.getCols() && B.getRows() == 1) {
    // Perform matrix addition with scalar value.
    Matrix<T> C(A.getRows(), A.getCols(), InitState::ZERO);
    for (size_t row = 0; row < A.getRows(); ++row) {
      for (size_t col = 0; col < A.getCols(); ++col) {
        C[row][col] = A[row][col] - B[0][col];
      }
    }
    return C;
  }
  if (A.getRows() == B.getRows() && B.getCols() == 1) {
    // Perform matrix addition with scalar value.
    Matrix<T> C(A.getRows(), A.getCols(), InitState::ZERO);
    for (size_t row = 0; row < A.getRows(); ++row) {
      for (size_t col = 0; col < A.getCols(); ++col) {
        C[row][col] = A[row][col] - B[row][0];
      }
    }
    return C;
  }

  // Check if matrices are in the same vectorspace.
  if (A.getRows() != B.getCols() || A.getCols() != B.getCols()) {
    throw std::invalid_argument("Matrices dimensions do not match for sub.");
  }
  // Perform matrix addition.
  Matrix<T> C(A.getRows(), A.getCols(), InitState::ZERO);
  for (size_t row = 0; row < A.getRows(); ++row) {
    for (size_t col = 0; col < A.getCols(); ++col) {
      C[row][col] = A[row][col] - B[row][col];
    }
  }
  return C;
}

// ____________________________________________________________________________
template <typename T> Matrix<T> dotElementWise(Matrix<T> &A, Matrix<T> &B) {
  // Check if matrices are in the same vectorspace.
  if (A.getRows() != B.getRows()) {
    throw std::invalid_argument(
        "Matrices dimensions do not match for element wise multiplication.");
  }

  // Perform element wise multiplication.
  Matrix<T> C(A.getRows(), B.getCols(), InitState::ZERO);
  for (size_t row = 0; row < A.getRows(); ++row) {
    for (size_t col = 0; col < B.getCols(); ++col) {
      C[row][col] += A[row][col] * B[row][col];
    }
  }
  return C;
}

// ____________________________________________________________________________
// Explicit instantiations (for linear algebra helper functions) for int, float.

template Matrix<int> dot<int>(Matrix<int> &A, Matrix<int> &B);
template Matrix<float> dot<float>(Matrix<float> &A, Matrix<float> &B);

template Matrix<int> add<int>(Matrix<int> &A, Matrix<int> &B);
template Matrix<float> add<float>(Matrix<float> &A, Matrix<float> &B);

template Matrix<int> sub<int>(Matrix<int> &A, Matrix<int> &B);
template Matrix<float> sub<float>(Matrix<float> &A, Matrix<float> &B);

template Matrix<int> dotElementWise<int>(Matrix<int> &A, Matrix<int> &B);
template Matrix<float> dotElementWise<float>(Matrix<float> &A,
                                             Matrix<float> &B);