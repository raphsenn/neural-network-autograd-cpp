
// Zero elements for diffrent types (int, float, double).

// Zero value templated struct.
template <typename T>
struct zero_value {
  static T value() { return T{}; }
};

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
struct random_value {
  static T value() { return T{}; }
};

template <>
struct random_value<int> {
  static int value() { return 0; }
};


