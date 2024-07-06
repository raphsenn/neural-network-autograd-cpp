
#pragma once

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

  Matrix(std::size_t rows, std::size_t cols, InitState state = InitState::RANDOM);
  ~Matrix();

  // Matrix access.
  T* operator[](std::size_t row);
  const T* operator[](std::size_t col) const;

  // Returns number of rows.
  std::size_t getRows() const;
  // Returns number of cols.
  std::size_t getCols() const;
  
  // Prints matrix (like a numpy matrix) in human readable format.
  void print();
};
