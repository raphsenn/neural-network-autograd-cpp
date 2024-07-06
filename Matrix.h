
#pragma once

#include <vector>
#include <cstddef>
#include <iostream>

// Matrix or neural network states.
enum class InitState {
  ZERO,
  RANDOM
};

// A simple matrix.
template <typename T> class Matrix {
private:
  // Rows, cols and matrix elements.
  std::size_t rows_;
  std::size_t cols_;
  T** matrix_;

  // Fills the matrix with zeros.
  void fillZeros();
  // Fills the matrix with random <T> values.
  void fillRandom();

public:
  // Constructor.
  Matrix(std::size_t rows, std::size_t cols, InitState state = InitState::RANDOM);

  // Copy-Constructor for Matrix<T>.
  Matrix(const Matrix<T>& matrix);

  // Copy-Constructor for std::vector<std::vector<T>> (2D-Vector).
  Matrix(const std::vector<std::vector<T>>& matrix);

  // Destructor. 
  ~Matrix();

  // Copy-Assignment operator.
  void operator=(const Matrix<T>& other);
  
  // Copy-Assignment operator for std::vector<std::vector<T>> (2D-Vector).
  void operator=(const std::vector<std::vector<T>> other);

  // Matrix access (for testing purpose only).
  T* operator[](const std::size_t row);
  const T* operator[](const std::size_t col) const;

  // ____________________________________________________________________________
  // Linear Algebra
 
  // Performs matrix addition.
  void add(const Matrix<T>& other) const;

  // Performs matrix multiplication. 
  void dot(const Matrix<T>& other) const;
 

  // Returns number of rows.
  std::size_t getRows() const;
  // Returns number of cols.
  std::size_t getCols() const;
  
  // Prints matrix (like a numpy matrix) in human readable format.
  void print();
};
