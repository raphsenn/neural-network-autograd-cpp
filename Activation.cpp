
#include "./Activation.h"

template <typename T>
Matrix<T> linear(Matrix<T>& X) { return X; }

template <typename T>
Matrix<T> linear_derivative(Matrix<T>& X) { return Matrix<T>(X.getRows(), X.getCols(), InitState::ONES); }

// Explicit instantiations for float.
template Matrix<float> linear<float>(Matrix<float>& X);
template Matrix<float> linear_derivative<float>(Matrix<float>& X);