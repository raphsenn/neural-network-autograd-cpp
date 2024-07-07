
#pragma once

#include <vector>
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
  T** matrix_;

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
  Matrix(const Matrix<T>& matrix);

  // Copy-Constructor for std::vector<std::vector<T>> (2D-Vector).
  Matrix(const std::vector<std::vector<T>>& matrix);

  // Move-Constructor for Matrix<T>.
  Matrix(Matrix<T>&& matrix);

  // Destructor. 
  ~Matrix();
  
  // ____________________________________________________________________________
  // Operators:
  // ____________________________________________________________________________
 
  // Copy-Assignment operator.
  Matrix<T> &operator=(const Matrix<T>& other);
  
  // Copy-Assignment operator for std::vector<std::vector<T>> (2D-Vector).
  Matrix<T> &operator=(const std::vector<std::vector<T>> other);

  // Move-Assignment operator for Matrix<T>.
  Matrix<T> &operator=(Matrix<T>&& other);

  // Matrix access (for testing purpose only).
  T* operator[](const std::size_t row);
  const T* operator[](const std::size_t col) const;

  // Check if two matrices are the same.
  bool operator==(const Matrix<T> other) const;

  // ____________________________________________________________________________
  // Linear Algebra methods:
  // ____________________________________________________________________________
 
  // Performs matrix addition.
  void add(const Matrix<T>& other);

  // Performs matrix subtraction.
  void sub(const Matrix<T>& other);

  // Performs matrix multiplication. 
  // m x n * n x k = m x k
  void dot(const Matrix<T>& other);
  
  // Transpose the matrix.
  // m x n -> n x m
  void transpose();

  // matrix[i][j] = matrix[i][j]) if matrix[i][j] >= inf
  // else inf, for all i, j.
  void maximum(T inf);

   // Sums all entry to one scalar. 
  T sum() const;
  
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