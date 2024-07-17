
#include <gtest/gtest.h>
#include <cmath>

#include "./NeuralNetwork.h"

// ____________________________________________________________________________
// Compare two float values within an epsilon range.
bool areAlmostEqual(float a, float b, float epsilon = 0.1f) {
  return std::fabs(a - b) < epsilon;
}

// ____________________________________________________________________________
TEST(LearnsToHalfeRealNumbers, NeuralNetwork) {
  // Simplest Neural Network, no complexity.
  // This neural network learns how to halve a number. 
  Matrix<float> X_train = std::vector<std::vector<float>>({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}});
  Matrix<float> y_train = std::vector<std::vector<float>>({{0.0f}, {0.5f}, {1.0f}, {1.5f}, {2.0f}, {2.5f}, {3.0f}, {3.5f}});
  NeuralNetwork<float> nn(std::vector<size_t>({1, 1}), std::vector<Activation>({Activation::linear}), 0.01f, InitState::RANDOM);
  nn.train(X_train, y_train, 1, 0.1f, 10000, false);
  Matrix<float> X_test = std::vector<std::vector<float>>({{-3}, {-2}, {-1}, {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}});
  Matrix<float> y_out = nn.act(X_test);
  ASSERT_EQ(areAlmostEqual(y_out[0][0], -1.5f), true);
  ASSERT_EQ(areAlmostEqual(y_out[1][0], -1.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[2][0], -0.5f), true);
  ASSERT_EQ(areAlmostEqual(y_out[3][0], 0.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[4][0], 0.5f), true);
  ASSERT_EQ(areAlmostEqual(y_out[5][0], 1.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[6][0], 1.5f), true);
  ASSERT_EQ(areAlmostEqual(y_out[7][0], 2.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[8][0], 2.5f), true);
  ASSERT_EQ(areAlmostEqual(y_out[9][0], 3.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[10][0], 3.5f), true);
  ASSERT_EQ(areAlmostEqual(y_out[11][0], 4.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[12][0], 4.5f), true);
}

// ____________________________________________________________________________
TEST(AndGate, NeuralNetwork) {
  Matrix<float> X_train = std::vector<std::vector<float>>({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
  Matrix<float> y_train = std::vector<std::vector<float>>({{0}, {0}, {0}, {1}});
  NeuralNetwork<float> nn(std::vector<size_t>({2, 1}), std::vector<Activation>({Activation::sigmoid}), 0.1f, InitState::RANDOM);
  nn.train(X_train, y_train,1, 0.1f, 10000, false);
  Matrix<float> y_out = nn.act(X_train);
  ASSERT_EQ(areAlmostEqual(y_out[0][0], 0.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[1][0], 0.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[2][0], 0.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out[3][0], 1.0f), true);
}

// ____________________________________________________________________________
TEST(OrGate, NeuralNetwork) {
  Matrix<float> X_train = std::vector<std::vector<float>>({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
  Matrix<float> y_train = std::vector<std::vector<float>>({{0}, {1}, {1}, {1}});
  NeuralNetwork<float> orGate(std::vector<size_t>({2, 1}), std::vector<Activation>({Activation::sigmoid}), 0.1f, InitState::RANDOM);
  orGate.train(X_train, y_train,1, 0.1f, 10000, false);
  Matrix<float> y_out = orGate.act(X_train);
  ASSERT_EQ(areAlmostEqual(y_out.getValue(0, 0), 0.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out.getValue(1, 0), 1.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out.getValue(2, 0), 1.0f), true);
  ASSERT_EQ(areAlmostEqual(y_out.getValue(3, 0), 1.0f), true);
}

// ____________________________________________________________________________
TEST(XorGate, NeuralNetwork) {
  // This neural network learns how to solve the XOR-Gate problem. 
  Matrix<float> X_train = std::vector<std::vector<float>>({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
  Matrix<float> y_train = std::vector<std::vector<float>>({{0}, {1}, {1}, {0}});
  NeuralNetwork<float> xorGate(std::vector<size_t>({2, 4, 1}), std::vector<Activation>({Activation::sigmoid, Activation::sigmoid}), 0.88f, InitState::RANDOM);
  xorGate.train(X_train, y_train,1, 0.1f, 10000, false);
  Matrix<float> y_out = xorGate.act(X_train);
  ASSERT_EQ(areAlmostEqual(y_out[0][0], 0.0f, 0.2f), true);
  ASSERT_EQ(areAlmostEqual(y_out[1][0], 1.0f, 0.2f), true);
  ASSERT_EQ(areAlmostEqual(y_out[2][0], 1.0f, 0.2f), true);
  ASSERT_EQ(areAlmostEqual(y_out[3][0], 0.0f, 0.2f), true);
}