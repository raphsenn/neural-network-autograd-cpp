
#pragma once

#include "./Matrix.h"

// ____________________________________________________________________________
// LINEAR
template <typename T>
Matrix<T> linear(Matrix<T>& X);

// linear_derivative(X) = 1
template <typename T>
Matrix<T> linear_derivative(Matrix<T>& X);

// ____________________________________________________________________________
// RELU
template <typename T>
Matrix<T> relu(Matrix<T>& X);

template <typename T>
Matrix<T> relu_derivative(Matrix<T>& X);

// ____________________________________________________________________________
// UNIT-STEP 
template <typename T>
Matrix<T> step(Matrix<T>& X);

template <typename T>
Matrix<T> step_derivative(Matrix<T>& X);

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
Matrix<T> sigmoid(Matrix<T>& X);

template <typename T>
Matrix<T> sigmoid_derivative(Matrix<T>& X);

// ____________________________________________________________________________
// TANH
template <typename T>
Matrix<T> tanh(Matrix<T>& X);

template <typename T>
Matrix<T> tanh_derivative(Matrix<T>& X);