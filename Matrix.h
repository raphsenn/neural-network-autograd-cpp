
#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <iostream>

// Different matrix states.
enum class InitState {
  ZERO,
  RANDOM,
  ONES,
  EMPTY
};

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
  Matrix(std::size_t rows, std::size_t cols, InitState state = InitState::RANDOM);

  // Copy-Constructor for Matrix<T>.
  Matrix(const Matrix<T>& matrix) = default;

  // Copy-Constructor for std::vector<std::vector<T>> (2D-Vector).
  Matrix(const std::vector<std::vector<T>>& matrix);

  // Move-Constructor for Matrix<T>.
  Matrix(Matrix<T>&& matrix) = default;

  // Destructor. 
  ~Matrix() = default;
  
  // ____________________________________________________________________________
  // Operators:
  // ____________________________________________________________________________
 
  // Copy-Assignment operator.
  Matrix<T> &operator=(const Matrix<T>& other) = default;
  
  // Copy-Assignment operator for std::vector<std::vector<T>> (2D-Vector).
  Matrix<T> &operator=(const std::vector<std::vector<T>> other);

  // Move-Assignment operator for Matrix<T>.
  Matrix<T> &operator=(Matrix<T>&& other) = default;

  // Matrix access (for testing purpose only).
  std::vector<T>& operator[](const std::size_t row);
  const std::vector<T>& operator[](const std::size_t col) const;

  // Check if two matrices are the same.
  bool operator==(const Matrix<T> other) const;

  // ____________________________________________________________________________
  // Linear Algebra methods:
  // ____________________________________________________________________________
 
  // Performs matrix addition.
  Matrix<T> add(const Matrix<T>& other);

  // Performs matrix subtraction.
  Matrix<T> sub(const Matrix<T>& other);

  // Performs matrix multiplication. 
  // m x n * n x k = m x k
  Matrix<T> dot(const Matrix<T>& other);
  
  // Element wise multiplication.
  // m x n * m x n = m x n
  Matrix<T> dotElementWise(const Matrix<T>& other);
  
  // Transpose the matrix.
  // m x n -> n x m
  Matrix<T> transpose();

  // Scalar multiplication.
  Matrix<T> scalMul(T scalar);
  
  // matrix[i][j] = matrix[i][j]) if matrix[i][j] >= inf
  // else inf, for all i, j.
  void maximum(T inf);

   // Sums all entry to one scalar. 
  Matrix<T> sum() const;
  
  // ____________________________________________________________________________
  // More methods (public):
  // ____________________________________________________________________________
 
  // Returns number of rows.
  std::size_t getRows() const;
  
  // Returns number of cols.
  std::size_t getCols() const;
  
  // Prints matrix (like a numpy matrix) in human readable format.
  void print();
};