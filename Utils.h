
#include <random>
#include <type_traits>

// Zero elements for diffrent types (int, float, double).

template <typename T>
struct zero_value {};

// Zero value for int.
template <>
struct zero_value<int> {
  static int value() { return 0; }
};

// Zero value for float.
template <>
struct zero_value<float> {
  static int value() { return 0.0f; }
};

// Zero value for double.
template <>
struct zero_value<double> {
  static int value() { return 0.0; }
};


// Random elements for diffrent types (int, float, double).

template <typename T>
struct random_value {};

// Create a random integer number.
template <>
struct random_value<int> {
  static int value() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> distribution(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    return distribution(gen);
  }
};

// Create a random float number between 0 and 1.
template <>
struct random_value<float> {
  static float value() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(gen);
  }
};

// Create a random double number between 0 and 1.
template <>
struct random_value<double> {
  static float value() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(gen);
  }
};