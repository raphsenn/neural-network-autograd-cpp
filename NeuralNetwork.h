
#include "./Matrix.h"
#include "./Activation.h"

template <typename S=float>
class NeuralNetwork {

private:
  void forward();
  void backward();

public:
  NeuralNetwork(std::vector<int> layers, std::vector<std::string> activations_functions, float learning_rate=0.1f, InitState state = InitState::RANDOM);

  void train();

};

