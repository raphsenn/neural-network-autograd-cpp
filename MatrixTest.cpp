
#include <gtest/gtest.h>
#include <string>

#include "./Matrix.h"


// ____________________________________________________________________________
// Constructor and Copy-Assignment:
// ____________________________________________________________________________

// ____________________________________________________________________________
TEST(ConstructorInitStateNULL, Matrix) {
  Matrix<float> A(1, 1, InitState::ZERO);
  ASSERT_EQ(A.getRows(), size_t(1));
  ASSERT_EQ(A.getCols(), size_t(1));
  ASSERT_FLOAT_EQ(A[0][0], 0.0f);

  Matrix<float> B(2, 3, InitState::ZERO);
  ASSERT_EQ(B.getRows(), size_t(2));
  ASSERT_EQ(B.getCols(), size_t(3));
  ASSERT_FLOAT_EQ(B[0][0], 0.0f);
  ASSERT_FLOAT_EQ(B[0][1], 0.0f);
  ASSERT_FLOAT_EQ(B[0][2], 0.0f);
  ASSERT_FLOAT_EQ(B[1][0], 0.0f);
  ASSERT_FLOAT_EQ(B[1][1], 0.0f);
  ASSERT_FLOAT_EQ(B[1][2], 0.0f);
  
  Matrix<double> C(3, 3, InitState::ZERO);
  ASSERT_EQ(C.getRows(), size_t(3));
  ASSERT_EQ(C.getCols(), size_t(3));
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

// ____________________________________________________________________________
TEST(ConstructorThrowInvalidArgument, Matrix) {
  EXPECT_THROW(Matrix<float> A(0, 1), std::invalid_argument);
  EXPECT_THROW(Matrix<float> B(1, 0), std::invalid_argument);
}

// ____________________________________________________________________________
TEST(CopyConstructorVector, Matrix) {
  std::vector<std::vector<int>> vectorA = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<int> A(vectorA);
  ASSERT_EQ(A[0][0], 1);
  ASSERT_EQ(A[0][1], 2);
  ASSERT_EQ(A[0][2], 3);
  ASSERT_EQ(A[1][0], 4);
  ASSERT_EQ(A[1][1], 5);
  ASSERT_EQ(A[1][2], 6);
  ASSERT_EQ(A[2][0], 7);
  ASSERT_EQ(A[2][1], 8);
  ASSERT_EQ(A[2][2], 9);

  std::vector<std::vector<float>> vectorB = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}};
  Matrix<float> B(vectorB);
  ASSERT_FLOAT_EQ(B[0][0], 0.1f);
  ASSERT_FLOAT_EQ(B[0][1], 0.2f);
  ASSERT_FLOAT_EQ(B[0][2], 0.3f);
  ASSERT_FLOAT_EQ(B[1][0], 0.4f);
  ASSERT_FLOAT_EQ(B[1][1], 0.5f);
  ASSERT_FLOAT_EQ(B[1][2], 0.6f);
  ASSERT_FLOAT_EQ(B[2][0], 0.7f);
  ASSERT_FLOAT_EQ(B[2][1], 0.8f);
  ASSERT_FLOAT_EQ(B[2][2], 0.9f);
}

// ____________________________________________________________________________
TEST(CopyAssignmentVector, Matrix) {
  std::vector<std::vector<int>> vectorA = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<int> A = vectorA;
  ASSERT_EQ(A[0][0], 1);
  ASSERT_EQ(A[0][1], 2);
  ASSERT_EQ(A[0][2], 3);
  ASSERT_EQ(A[1][0], 4);
  ASSERT_EQ(A[1][1], 5);
  ASSERT_EQ(A[1][2], 6);
  ASSERT_EQ(A[2][0], 7);
  ASSERT_EQ(A[2][1], 8);
  ASSERT_EQ(A[2][2], 9);

  std::vector<std::vector<float>> vectorB = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}};
  Matrix<float> B = vectorB;
  ASSERT_FLOAT_EQ(B[0][0], 0.1f);
  ASSERT_FLOAT_EQ(B[0][1], 0.2f);
  ASSERT_FLOAT_EQ(B[0][2], 0.3f);
  ASSERT_FLOAT_EQ(B[1][0], 0.4f);
  ASSERT_FLOAT_EQ(B[1][1], 0.5f);
  ASSERT_FLOAT_EQ(B[1][2], 0.6f);
  ASSERT_FLOAT_EQ(B[2][0], 0.7f);
  ASSERT_FLOAT_EQ(B[2][1], 0.8f);
  ASSERT_FLOAT_EQ(B[2][2], 0.9f);
}

// ____________________________________________________________________________
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

// ____________________________________________________________________________
// Linear Algebra Operations:
// ____________________________________________________________________________ 

// ____________________________________________________________________________
TEST(Addition, Matrix) {
  std::vector<std::vector<int>> vec1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<int> A = vec1;
  Matrix<int> B = vec1;
  A.add(B);
  ASSERT_EQ(A[0][0], 2);
  ASSERT_EQ(A[0][1], 4);
  ASSERT_EQ(A[0][2], 6);
  ASSERT_EQ(A[1][0], 8);
  ASSERT_EQ(A[1][1], 10);
  ASSERT_EQ(A[1][2], 12);
  ASSERT_EQ(A[2][0], 14);
  ASSERT_EQ(A[2][1], 16);
  ASSERT_EQ(A[2][2], 18);

  std::vector<std::vector<int>> vec2 = {{1, 2}, {3, 4}};
  std::vector<std::vector<int>> vec3 = {{-1, -2}, {-3, -4}};
  Matrix<int> C = vec2;
  Matrix<int> D = vec3;
  C.add(D);
  ASSERT_EQ(C[0][0], 0);
  ASSERT_EQ(C[0][1], 0);
  ASSERT_EQ(C[1][0], 0);
  ASSERT_EQ(C[0][1], 0);
}



