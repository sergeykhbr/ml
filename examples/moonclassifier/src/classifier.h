#pragma once

#include <random>

// Input parameters:
const float IN_DATA_DISTRIBUTION = 0.2f;

// Architecture Parameters
const int INPUT_DIM = 2;   // (x, y) coordinates
const int HIDDEN_DIM = 4;  // Tiny layer so we can see every single number
const int OUTPUT_DIM = 2;  // 2 classes: Red (0) or Blue (1)

class MoonClassifier {
 public:
    explicit MoonClassifier(std::mt19937 &gen);

    void trainStep(float x_val, float y_val, int target_label, bool print_debug);
    float forwardPass(float x, float y);

 private:
    // Model Parameters (Weights and Biases flattened into 1D memory)
    float W1[INPUT_DIM * HIDDEN_DIM];
    float B1[HIDDEN_DIM];
    float W2[HIDDEN_DIM * OUTPUT_DIM];
    float B2[OUTPUT_DIM];

    float learning_rate = 0.05f;

};
