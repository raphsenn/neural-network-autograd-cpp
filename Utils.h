
#include <random>
#include <type_traits>
#include <cmath>

// ____________________________________________________________________________
template <typename T>
struct value {};

// ____________________________________________________________________________
// INT:
template <>
struct value<int> {
  static int zero() { return 0; }
  static int one() { return 1; }
  static int random() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> distribution(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    return distribution(gen);
  }
  static int e(int x) { return std::exp(x); }

};

// ____________________________________________________________________________
// FLOAT:
template <>
struct value<float> {
  static float zero() { return 0.0f; }
  static float one() { return 1.0f; }
  static float random() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(gen);
  }
  static float e(float x) { return std::exp(x); }
};

// ____________________________________________________________________________
// DOUBLE:
template <>
struct value<double> {
  static double zero() { return 0.0; }
  static double one() { return 0.0; }
  static double random() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(gen);
  }
  static double e(double x) { return std::exp(x); }

};