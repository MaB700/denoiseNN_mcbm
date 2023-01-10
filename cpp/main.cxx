#include <chrono>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
class MLP {
  public:
  MLP(std::vector<int> layers)
      : layers_(layers) {
    // Initialize weights and biases with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (int i = 0; i < layers_.size() - 1; ++i) {
      int input_size = layers_[i];
      int output_size = layers_[i + 1];
      weights_.emplace_back(output_size, std::vector<T>(input_size));
      biases_.emplace_back(output_size);
      for (int j = 0; j < output_size; ++j) {
        biases_[i][j] = dist(gen);
        for (int k = 0; k < input_size; ++k) {
          weights_[i][j][k] = dist(gen);
        }
      }
    }
  }

  std::vector<T> forward(std::vector<T> input) {
    std::vector<T> activations = input;
    for (int i = 0; i < layers_.size() - 1; ++i) {
      activations = forward_layer(activations, weights_[i], biases_[i]);
    }
    return activations;
  }

  private:
  std::vector<int> layers_;
  std::vector<std::vector<std::vector<T>>> weights_;
  std::vector<std::vector<T>> biases_;

  std::vector<T> forward_layer(const std::vector<T>& input,
                               const std::vector<std::vector<T>>& weights,
                               const std::vector<T>& biases) {
    std::vector<T> output(weights.size());
    for (int i = 0; i < weights.size(); ++i) {
      T sum = 0;
      for (int j = 0; j < weights[i].size(); ++j) {
        sum += weights[i][j] * input[j];
      }
      sum += biases[i];
      output[i] = sum;
    }
    return output;
  }
};

int main() {
  MLP<float> mlp({32, 1});
  // float vector with 10 random elements
  std::vector<float> input(100);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  for (int i = 0; i < 100; ++i) {
    input[i] = dist(gen);
  }  
  // call ml.forward(input) 1000 times are measure the time
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10000; ++i) {
    mlp.forward(input);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  // print duration in ms
  std::cout << "Elapsed time: " << elapsed.count() /10000. * (1000*1000) << " us" << std::endl;

  //   std::vector<float> output = mlp.forward(input);
  //   std::cout << "Output: " << output[0] << std::endl;
  return 0;
}
