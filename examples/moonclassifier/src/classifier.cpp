/*
 *  Copyright 2026 Sergei Khabarov, sergeykhbr@gmail.com
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "classifier.h"

MoonClassifier::MoonClassifier(std::mt19937 &gen) {
    float scale1 = std::sqrt(2.0f / INPUT_DIM);
    std::normal_distribution<float> dist1(0.0f, scale1);
    for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; ++i) {
        W1[i] = dist1(gen);
    }
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        B1[i] = 0.0f;
    }

    float scale2 = std::sqrt(2.0f / HIDDEN_DIM);
    std::normal_distribution<float> dist2(0.0f, scale2);
    for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; ++i) {
        W2[i] = dist2(gen);
    }
    for (int i = 0; i < OUTPUT_DIM; ++i) {
        B2[i] = 0.0f;
    }
}

float MoonClassifier::forwardPass(float *IN, float *OUT) {
    // Hidden Layer: Z1 = X * W1 + B1
    for (int j = 0; j < HIDDEN_DIM; ++j) {
        Z1[j] = 0.0f;
        for (int i = 0; i < INPUT_DIM; ++i) {
            Z1[j] += IN[i] * W1[i * HIDDEN_DIM + j];
        }
        Z1[j] += B1[j];
    }

    // Activation: A1 = ReLU(Z1)
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        A1[i] = std::max(0.0f, Z1[i]);
    }

    // Output Layer: Z2 = A1 * W2 + B2
    for (int j = 0; j < OUTPUT_DIM; ++j) {
        Z2[j] = 0.0f;
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            Z2[j] += A1[i] * W2[i * OUTPUT_DIM + j];
        }
        Z2[j] += B2[j];
    }

    // Activation: A2 = Softmax(Z2)
    float max_z = std::max(Z2[0], Z2[1]);       // Stability trick
    float sum_exp = std::exp(Z2[0] - max_z) + std::exp(Z2[1] - max_z);
    OUT[0] = std::exp(Z2[0] - max_z) / sum_exp;
    OUT[1] = std::exp(Z2[1] - max_z) / sum_exp;  // probability

    return OUT[1];
}

void MoonClassifier::trainStep(DataPoint *datapoint) {
    // --- 1. FORWARD PASS ---
    float X[INPUT_DIM] = {datapoint->x, datapoint->y};
    float A2[OUTPUT_DIM];

    forwardPass(X, A2);

    // Calculate Target Vector Y (One-hot mapping)
    float Y[OUTPUT_DIM] = {0.0f};
    Y[datapoint->label] = 1.0f;

    // Calculate loss for tracking: Cross-Entropy = -sum(Y * log(A2))
    float loss = -std::log(std::max(A2[datapoint->label], 1e-7f));

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
}

