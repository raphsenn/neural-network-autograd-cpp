
#include <gtest/gtest.h>
#include <string>

#include "./Matrix.h"

TEST(ConstructorInitStateNULL, Matrix) {
  Matrix<float> A(1, 1, InitState::ZERO);
  ASSERT_EQ(A.getRows(), 1);
  ASSERT_EQ(A.getCols(), 1);
  ASSERT_FLOAT_EQ(A[0][0], 0.0f);

  Matrix<float> B(2, 3, InitState::ZERO);
  ASSERT_EQ(B.getRows(), 2);
  ASSERT_EQ(B.getCols(), 3);
  ASSERT_FLOAT_EQ(B[0][0], 0.0f);
  ASSERT_FLOAT_EQ(B[0][1], 0.0f);
  ASSERT_FLOAT_EQ(B[0][2], 0.0f);
  ASSERT_FLOAT_EQ(B[1][0], 0.0f);
  ASSERT_FLOAT_EQ(B[1][1], 0.0f);
  ASSERT_FLOAT_EQ(B[1][2], 0.0f);
  
  Matrix<double> C(3, 3, InitState::ZERO);
  ASSERT_EQ(C.getRows(), 3);
  ASSERT_EQ(C.getCols(), 3);
  ASSERT_DOUBLE_EQ(C[0][0], 0.0);
  ASSERT_DOUBLE_EQ(C[0][1], 0.0);
  ASSERT_DOUBLE_EQ(C[0][2], 0.0);
  ASSERT_DOUBLE_EQ(C[1][0], 0.0);
  ASSERT_DOUBLE_EQ(C[1][1], 0.0);
  ASSERT_DOUBLE_EQ(C[1][2], 0.0);
  ASSERT_DOUBLE_EQ(C[2][0], 0.0);
  ASSERT_DOUBLE_EQ(C[2][1], 0.0);
  ASSERT_DOUBLE_EQ(C[2][2], 0.0);
}

TEST(ConstructorThrowInvalidArgument, Matrix) {
  EXPECT_THROW(Matrix<float> A(0, 1), std::invalid_argument);
  EXPECT_THROW(Matrix<float> B(1, 0), std::invalid_argument);
}

TEST(Print, Matrix) {
  Matrix<float> A(1, 1, InitState::ZERO);
  Matrix<float> B(3, 3, InitState::ZERO);
  
  // Test printing. 
  testing::internal::CaptureStdout();
  A.print();
  std::string A_as_string = testing::internal::GetCapturedStdout();
  EXPECT_EQ(A_as_string, "matrix([[0]])\n");

  testing::internal::CaptureStdout();
  B.print();
  std::string B_as_string = testing::internal::GetCapturedStdout();
  EXPECT_EQ(B_as_string, "matrix([[0, 0, 0],\n[0, 0, 0],\n[0, 0, 0]])\n");

}

