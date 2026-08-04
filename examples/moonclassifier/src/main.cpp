#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Architecture Parameters
const int INPUT_DIM = 2;   // (x, y) coordinates
const int HIDDEN_DIM = 4;  // Tiny layer so we can see every single number
const int OUTPUT_DIM = 2;  // 2 classes: Red (0) or Blue (1)

// Structures to hold our dataset
struct DataPoint {
    float x;
    float y;
    int label; // 0 for Red, 1 for Blue
};

class MoonClassifier {
private:
    // Model Parameters (Weights and Biases flattened into 1D memory)
    float W1[INPUT_DIM * HIDDEN_DIM];
    float B1[HIDDEN_DIM];
    float W2[HIDDEN_DIM * OUTPUT_DIM];
    float B2[OUTPUT_DIM];

    float learning_rate = 0.05f;

public:
    MoonClassifier() {
        // Initialize weights using standard Mersenne Twister and Xavier scaling
        std::random_device rd;
        std::mt19937 gen(rd());
        
        float scale1 = std::sqrt(2.0f / INPUT_DIM);
        std::normal_distribution<float> dist1(0.0f, scale1);
        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; ++i) W1[i] = dist1(gen);
        for (int i = 0; i < HIDDEN_DIM; ++i) B1[i] = 0.0f;

        float scale2 = std::sqrt(2.0f / HIDDEN_DIM);
        std::normal_distribution<float> dist2(0.0f, scale2);
        for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; ++i) W2[i] = dist2(gen);
        for (int i = 0; i < OUTPUT_DIM; ++i) B2[i] = 0.0f;
    }

    void trainStep(float x_val, float y_val, int target_label, bool print_debug) {
        // --- 1. FORWARD PASS ---
        float X[INPUT_DIM] = {x_val, y_val};

        // Hidden Layer: Z1 = X * W1 + B1
        float Z1[HIDDEN_DIM] = {0.0f};
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            for (int i = 0; i < INPUT_DIM; ++i) {
                Z1[j] += X[i] * W1[i * HIDDEN_DIM + j];
            }
            Z1[j] += B1[j];
        }

        // Activation: A1 = ReLU(Z1)
        float A1[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            A1[i] = std::max(0.0f, Z1[i]);
        }

        // Output Layer: Z2 = A1 * W2 + B2
        float Z2[OUTPUT_DIM] = {0.0f};
        for (int j = 0; j < OUTPUT_DIM; ++j) {
            for (int i = 0; i < HIDDEN_DIM; ++i) {
                Z2[j] += A1[i] * W2[i * OUTPUT_DIM + j];
            }
            Z2[j] += B2[j];
        }

        // Activation: A2 = Softmax(Z2)
        float A2[OUTPUT_DIM];
        float max_z = std::max(Z2[0], Z2[1]); // Stability trick
        float sum_exp = std::exp(Z2[0] - max_z) + std::exp(Z2[1] - max_z);
        A2[0] = std::exp(Z2[0] - max_z) / sum_exp;
        A2[1] = std::exp(Z2[1] - max_z) / sum_exp;

        // Calculate Target Vector Y (One-hot mapping)
        float Y[OUTPUT_DIM] = {0.0f};
        Y[target_label] = 1.0f;

        // Calculate loss for tracking: Cross-Entropy = -sum(Y * log(A2))
        float loss = -std::log(std::max(A2[target_label], 1e-7f));

        // --- 2. BACKWARD PASS (THE CALCULUS) ---

        // Error at output layer: dZ2 = A2 - Y
        float dZ2[OUTPUT_DIM];
        dZ2[0] = A2[0] - Y[0];
        dZ2[1] = A2[1] - Y[1];

        // Error at hidden layer: dZ1 = (dZ2 * W2^T) * ReLU_derivative(Z1)
        float dZ1[HIDDEN_DIM] = {0.0f};
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            float error = 0.0f;
            for (int j = 0; j < OUTPUT_DIM; ++j) {
                error += dZ2[j] * W2[i * OUTPUT_DIM + j]; // Transposed matrix index
            }
            dZ1[i] = (Z1[i] > 0.0f) ? error : 0.0f; // ReLU derivative
        }

        // --- 3. GRADIENT DESCENT PARAMETER UPDATES ---

        // Update W2 and B2
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            for (int j = 0; j < OUTPUT_DIM; ++j) {
                W2[i * OUTPUT_DIM + j] -= learning_rate * (A1[i] * dZ2[j]);
            }
        }
        B2[0] -= learning_rate * dZ2[0];
        B2[1] -= learning_rate * dZ2[1];

        // Update W1 and B1
        for (int i = 0; i < INPUT_DIM; ++i) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                W1[i * HIDDEN_DIM + j] -= learning_rate * (X[i] * dZ1[j]);
            }
        }
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            B1[j] -= learning_rate * dZ1[j];
        }

        // Optional log output to observe numbers in real-time
        if (print_debug) {
            std::cout << "Loss: " << loss 
                      << " | Predict Red: " << A2[0] * 100 << "%"
                      << " Blue: " << A2[1] * 100 << "%"
                      << " | dZ2: [" << dZ2[0] << ", " << dZ2[1] << "]" << std::endl;
        }
    }
};

int main() {
    // Generate Moon Dataset
    std::vector<DataPoint> dataset;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);

    // Create Red Moon coordinates along a curve
    for (int i = 0; i < 500; ++i) {
        float theta = (i / 500.0f) * 3.14159f;
        dataset.push_back({std::cos(theta) + dist(gen), std::sin(theta) + dist(gen), 0});
    }
    // Create Blue Moon coordinates shifted downward/sideways
    for (int i = 0; i < 500; ++i) {
        float theta = (i / 500.0f) * 3.14159f;
        dataset.push_back({1.0f - std::cos(theta) + dist(gen), 0.5f - std::sin(theta) + dist(gen), 1});
    }

    MoonClassifier model;

    std::cout << "--- Starting Training Optimization ---" << std::endl;
    // Run training over 20 epochs
    for (int epoch = 1; epoch <= 20; ++epoch) {
        // Shuffle the tracking points each epoch to maintain optimization stability
        std::shuffle(dataset.begin(), dataset.end(), gen);
        
        std::cout << "\n--- Epoch " << epoch << " ---" << std::endl;
        // Print the math state for the first point of the epoch to track progression
        model.trainStep(dataset[0].x, dataset[0].y, dataset[0].label, true);

        // Train silently on the rest of the points
        for (size_t i = 1; i < dataset.size(); ++i) {
            model.trainStep(dataset[i].x, dataset[i].y, dataset[i].label, false);
        }
    }

    return 0;
}
