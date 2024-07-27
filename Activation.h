
#pragma once

#include "./Matrix.h"

enum class Activation { linear, relu, step, softmax, sigmoid, tanh };

// ____________________________________________________________________________
// LINEAR
template <typename T> Matrix<T> linear(const Matrix<T> &X);

template <typename T> Matrix<T> linear_derivative(const Matrix<T> &X);

// ____________________________________________________________________________
// RELU
template <typename T> Matrix<T> relu(const Matrix<T> &X);

template <typename T> Matrix<T> relu_derivative(const Matrix<T> &X);

// ____________________________________________________________________________
// UNIT-STEP
template <typename T> Matrix<T> step(const Matrix<T> &X);

template <typename T> Matrix<T> step_derivative(const Matrix<T> &X);

// ____________________________________________________________________________
// SIGMOID
template <typename T> Matrix<T> sigmoid(const Matrix<T> &X);

template <typename T> Matrix<T> sigmoid_derivative(const Matrix<T> &X);

// ____________________________________________________________________________
// TANH
template <typename T> Matrix<T> tanh(const Matrix<T> &X);

template <typename T> Matrix<T> tanh_derivative(const Matrix<T> &X);

// ____________________________________________________________________________
// SOFTMAX
template <typename T> Matrix<T> softmax(const Matrix<T> &X);

template <typename T> Matrix<T> softmax_derivative(const Matrix<T> &X);

// ____________________________________________________________________________
// EXP
template <typename T> Matrix<T> exp(Matrix<T> &X);