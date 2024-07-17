
#pragma once

#include "./Matrix.h"

enum class Activation {
  linear,
  relu,
  step,
  softmax,
  maxout,
  sigmoid,
  tanh
};

// ____________________________________________________________________________
// LINEAR
template <typename T>
Matrix<T> linear(const Matrix<T>& X);

template <typename T>
Matrix<T> linear_derivative(const Matrix<T>& X);

// ____________________________________________________________________________
// RELU
template <typename T>
Matrix<T> relu(const Matrix<T>& X);

template <typename T>
Matrix<T> relu_derivative(const Matrix<T>& X);

// ____________________________________________________________________________
// UNIT-STEP 
template <typename T>
Matrix<T> step(const Matrix<T>& X);

template <typename T>
Matrix<T> step_derivative(const Matrix<T>& X);

// ____________________________________________________________________________
// SOFTMAX
template <typename T>
Matrix<T> softmax(Matrix<T>& X);

template <typename T>
Matrix<T> softmax_derivative(Matrix<T>& X);

// ____________________________________________________________________________
// MAXOUT 
template <typename T>
Matrix<T> maxout(Matrix<T>& X);

template <typename T>
Matrix<T> maxout_derivative(Matrix<T>& X);

// ____________________________________________________________________________
// SIGMOID 
template <typename T>
Matrix<T> sigmoid(const Matrix<T>& X);

template <typename T>
Matrix<T> sigmoid_derivative(const Matrix<T>& X);

// ____________________________________________________________________________
// TANH
template <typename T>
Matrix<T> tanh(Matrix<T>& X);

template <typename T>
Matrix<T> tanh_derivative(Matrix<T>& X);