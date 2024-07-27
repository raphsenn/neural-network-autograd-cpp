
#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

// Different matrix states.
enum class InitState { ZERO, RANDOM, ONES, EMPTY };

// A simple matrix.
template <typename T> class Matrix {
private:
  // ____________________________________________________________________________
  // Membervariables and Methods (private):
  // ____________________________________________________________________________

  // Rows, cols and matrix elements.
  std::size_t rows_;
  std::size_t cols_;
  // T** matrix_;
  std::vector<std::vector<T>> matrix_;

  // Fills the matrix with zeros.
  void fillZeros();

  // Fills the matrix with random <T> values.
  void fillRandom();

  // Fills the matrix with ones.
  void fillOnes();

public:
  // ____________________________________________________________________________
  // Constructors:
  // ____________________________________________________________________________

  // Constructor.
  Matrix(std::size_t rows, std::size_t cols,
         InitState state = InitState::RANDOM);

  // Copy-Constructor for Matrix<T>.
  Matrix(const Matrix<T> &matrix) = default;

  // Copy-Constructor for std::vector<std::vector<T>> (2D-Vector).
  Matrix(const std::vector<std::vector<T>> &matrix);

  // Move-Constructor for Matrix<T>.
  Matrix(Matrix<T> &&matrix) = default;

  // Destructor.
  ~Matrix() = default;

  // ____________________________________________________________________________
  // Operators:
  // ____________________________________________________________________________

  // Copy-Assignment operator.
  Matrix<T> &operator=(const Matrix<T> &other) = default;

  // Copy-Assignment operator for std::vector<std::vector<T>> (2D-Vector).
  Matrix<T> &operator=(const std::vector<std::vector<T>> other);

  // Move-Assignment operator for Matrix<T>.
  Matrix<T> &operator=(Matrix<T> &&other) = default;

  // Matrix access (for testing purpose only).
  std::vector<T> &operator[](const std::size_t row);
  const std::vector<T> &operator[](const std::size_t col) const;

  // Check if two matrices are the same.
  bool operator==(const Matrix<T> other) const;

  // ____________________________________________________________________________
  // Linear Algebra methods:
  // ____________________________________________________________________________

  // Performs matrix addition.
  Matrix<T> &add(const Matrix<T> &other);

  // Performs matrix subtraction.
  Matrix<T> sub(const Matrix<T> &other);

  // Performs matrix multiplication.
  // m x n * n x k = m x k
  Matrix<T> dot(const Matrix<T> &other);

  // Transpose the matrix.
  // m x n -> n x m
  Matrix<T> &transpose();

  // Same as transpose, just returns a new object.
  Matrix<T> transpose_copy();

  // Scalar multiplication.
  Matrix<T> scalMul(T scalar);

  // matrix[i][j] = matrix[i][j]) if matrix[i][j] >= inf, (inf = infimum).
  // else inf, for all i, j.
  void maximum(T inf);

  // Sums all entry to one scalar.
  Matrix<T> sum(bool axis = 0) const;

  // ____________________________________________________________________________
  // More methods (public):
  // ____________________________________________________________________________

  // Returns number of rows.
  std::size_t getRows() const;

  // Returns number of cols.
  std::size_t getCols() const;

  // Returns matrix data.
  std::vector<std::vector<T>> getData() const;

  // Returns of Value at row, col in matrix.
  T getValue(const size_t row, const size_t col) const;

  // Prints matrix (like a numpy matrix) in human readable format.
  void print() const;
};

// ____________________________________________________________________________
// Linear algebra functions:
// ____________________________________________________________________________

// Matrix multiplication.
template <typename T> Matrix<T> dot(const Matrix<T> &A, const Matrix<T> &B);

// Add to matrices.
template <typename T> Matrix<T> add(const Matrix<T> &A, const Matrix<T> &B);

// Subtract two matrices.
template <typename T> Matrix<T> sub(const Matrix<T> &A, const Matrix<T> &B);

// Dotproduct element wise:
// Matrix<int> A = {{1, 2}, {3, 4}};
// Matrix<int> B = {{1, 2}, {3, 4}};
// Matrix<int> RES = dotElementWise(A, B);
// RES = {{1 * 1, 2 * 2}, {3 * 3, 4 * 4}}
template <typename T>
Matrix<T> dotElementWise(const Matrix<T> &A, const Matrix<T> &B);

// Sums all entys in Matrix to one scalar.
template <typename T> T sum(Matrix<T> A);