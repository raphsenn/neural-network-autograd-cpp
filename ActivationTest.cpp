
#include <gtest/gtest.h>

#include "./Activation.h"

// ____________________________________________________________________________
TEST(Linear, Activation) {
  Matrix<float> X = std::vector<std::vector<float>>({{0.1f, 0.2f, 0.3f}, {0.01f, 0.02f, 0.03f}});
  Matrix<float> y = linear(X);
  EXPECT_EQ(X, y);
}

// ____________________________________________________________________________
TEST(LinearDerivative, Activation) {
  Matrix<float> X = std::vector<std::vector<float>>({{0.1f, 0.2f, 0.3f}, {0.01f, 0.02f, 0.03f}});
  Matrix<float> y = linear_derivative(X);
  EXPECT_EQ(y, Matrix<float>(2, 3, InitState::ONES));
}

// ____________________________________________________________________________
TEST(Relu, Activation) {
  Matrix<float> X = std::vector<std::vector<float>>({{-0.1f, 0.2f, -0.3f}, {0.01f, -0.02f, 0.03f}});
  Matrix<float> y_expect = std::vector<std::vector<float>>({{0.0f, 0.2f, 0.0f}, {0.01f, 0.0f, 0.03f}});
  relu(X); 
  EXPECT_EQ(X, y_expect);
}

// ____________________________________________________________________________
TEST(ReluDerivative, Activation) {
  Matrix<float> X = std::vector<std::vector<float>>({{0.1f, 0.2f, -0.3f}, {-0.01f, -0.02f, 0.03f}});
  Matrix<float> y_expect = std::vector<std::vector<float>>({{1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});
  relu_derivative(X);
  EXPECT_EQ(X, y_expect);
}

// ____________________________________________________________________________
TEST(Step, Activation) {
  Matrix<float> X = std::vector<std::vector<float>>({{-0.1f, 0.2f, -0.3f}, {0.01f, -0.02f, 0.03f}});
  Matrix<float> y_expect = std::vector<std::vector<float>>({{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}});
  step(X);
  EXPECT_EQ(X, y_expect);
}

// ____________________________________________________________________________
TEST(StepDerivative, Activation) {
  Matrix<float> X = std::vector<std::vector<float>>({{0.1f, 0.2f, -0.3f}, {-0.01f, -0.02f, 0.03f}});
  Matrix<float> y_expect = Matrix<float>(2, 3, InitState::ZERO); 
  step_derivative(X);
  EXPECT_EQ(X, y_expect);
}