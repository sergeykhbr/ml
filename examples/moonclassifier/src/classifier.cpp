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
#ifdef LAYER2_ENA
    float scale1 = std::sqrt(2.0f / INPUT_DIM);
    std::normal_distribution<float> dist1(0.0f, scale1);
    for (int i = 0; i < INPUT_DIM * HIDDEN1_DIM; ++i) {
        W1[i] = dist1(gen);
    }
    for (int i = 0; i < HIDDEN1_DIM; ++i) {
        B1[i] = 0.0f;
    }

    float scale2 = std::sqrt(2.0f / HIDDEN1_DIM);
    std::normal_distribution<float> dist2(0.0f, scale2);
    for (int i = 0; i < HIDDEN1_DIM * HIDDEN2_DIM; ++i) {
        W2[i] = dist2(gen);
    }
    for (int i = 0; i < HIDDEN2_DIM; ++i) {
        B2[i] = 0.0f;
    }

    float scale3 = std::sqrt(2.0f / HIDDEN2_DIM);
    std::normal_distribution<float> dist3(0.0f, scale3);
    for (int i = 0; i < HIDDEN2_DIM * OUTPUT_DIM; ++i) {
        W3[i] = dist3(gen);
    }
    for (int i = 0; i < OUTPUT_DIM; ++i) {
        B3[i] = 0.0f;
    }
#else
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
#endif
}

// OUT = IN * W1 + B1
void MoonClassifier::forwardLayer(float *IN, int isz,
                                  float *OUT, int osz,
                                  float *W,
                                  float *B) {
    for (int j = 0; j < osz; ++j) {
        OUT[j] = 0.0f;
        for (int i = 0; i < isz; ++i) {
            OUT[j] += IN[i] * W[i * osz + j];
        }
        OUT[j] += B[j];
    }
}

void MoonClassifier::backwardLayer(float *dIN, int isz,
                                   float *OUT, float *dOUT, int osz,
                                   float *W) {
    // Error at hidden layer: dOUT = (dIN * W^T) * ReLU_derivative(OUT)
    for (int i = 0; i < osz; ++i) {
        float error = 0.0f;
        for (int j = 0; j < isz; ++j) {
            error += dIN[j] * W[i * isz + j]; 
        }
#ifdef SIGMOID_ENA
        dOUT[i] = error * derivativeSigmoid(OUT[i]);
#else
        dOUT[i] = error * derivativeReLU(OUT[i]);
#endif
    }
}

float MoonClassifier::ReLU(float IN) {
    return std::max(0.0f, IN);
}

float MoonClassifier::derivativeReLU(float IN) {
    return IN > 0.0f ? 1.0f : 0.0f;
}

float MoonClassifier::Sigmoid(float IN) {
    return 1.0f / (1.0f + std::exp(-IN));
}

float MoonClassifier::derivativeSigmoid(float IN) {
    float act = Sigmoid(IN);
    return act * (1.0f - act);
}


void MoonClassifier::activation(float *IN, float *OUT, int sz) {
    for (int i = 0; i < sz; ++i) {
#ifdef SIGMOID_ENA
        OUT[i] = Sigmoid(IN[i]);
#else
        OUT[i] = ReLU(IN[i]);
#endif
    }
}

void MoonClassifier::activationSoftmax(float *IN, float *OUT, int sz) {
    float m = IN[0];
    float sum_exp = 0.0f;
    for (int i = 1; i < sz; i++) {
        m = std::max(m, IN[i]);       // Stability trick
    }
    for (int i = 0; i < sz; i++) {
        sum_exp += std::exp(IN[i] - m);
    }
    for (int i = 0; i < sz; i++) {
        OUT[i] = std::exp(IN[i] - m) / sum_exp;
    }
}

float MoonClassifier::forwardPass(float *IN, float *OUT) {
#ifdef LAYER2_ENA
    forwardLayer(IN, INPUT_DIM,
                 Z1, HIDDEN1_DIM,
                 W1,
                 B1);
    activation(Z1, A1, HIDDEN1_DIM);

    forwardLayer(A1, HIDDEN1_DIM,
                 Z2, HIDDEN2_DIM,
                 W2,
                 B2);
    activation(Z2, A2, HIDDEN2_DIM);

    forwardLayer(A2, HIDDEN2_DIM,
                 Z3, OUTPUT_DIM,
                 W3,
                 B3);
    activationSoftmax(Z3, OUT, OUTPUT_DIM);
#else
    forwardLayer(IN, INPUT_DIM,
                 Z1, HIDDEN_DIM,
                 W1,
                 B1);
    activation(Z1, A1, HIDDEN_DIM);

    forwardLayer(A1, HIDDEN_DIM,
                 Z2, OUTPUT_DIM,
                 W2,
                 B2);
    activationSoftmax(Z2, OUT, OUTPUT_DIM);
    /*
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
    */
#endif

    return OUT[1];
}

void MoonClassifier::trainStep(DataPoint *datapoint) {
    // --- 1. FORWARD PASS ---
    float X[INPUT_DIM] = {datapoint->x, datapoint->y};
#ifdef LAYER2_ENA
    float A3[OUTPUT_DIM];

    forwardPass(X, A3);
    // Calculate loss for tracking: Cross-Entropy = -sum(Y * log(A3))
    float loss = -std::log(std::max(A3[datapoint->label], 1e-7f));
#else
    float A2[OUTPUT_DIM];

    forwardPass(X, A2);
    // Calculate loss for tracking: Cross-Entropy = -sum(Y * log(A2))
    float loss = -std::log(std::max(A2[datapoint->label], 1e-7f));
#endif

    // Calculate Target Vector Y (One-hot mapping)
    float Y[OUTPUT_DIM] = {0.0f};
    Y[datapoint->label] = 1.0f;


    // 2. BACKWARD PASS
#ifdef LAYER2_ENA
    float dZ3[OUTPUT_DIM];
    for (int i = 0; i < OUTPUT_DIM; i++) {
        dZ3[i] = A3[i] - Y[i];
    }

    float dZ2[HIDDEN2_DIM];
    backwardLayer(dZ3, OUTPUT_DIM, Z2, dZ2, HIDDEN2_DIM, W3);

    float dZ1[HIDDEN1_DIM];
    backwardLayer(dZ2, HIDDEN2_DIM, Z1, dZ1, HIDDEN1_DIM, W2);

    // 3. GRADIENT DESCENT PARAMETER UPDATES
    for (int i = 0; i < HIDDEN2_DIM; ++i) {
        for (int j = 0; j < OUTPUT_DIM; ++j) {
            W3[i * OUTPUT_DIM + j] -= learning_rate * (A2[i] * dZ3[j]);
        }
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
        B3[i] -= learning_rate * dZ3[i];
    }

    for (int i = 0; i < HIDDEN1_DIM; ++i) {
        for (int j = 0; j < HIDDEN2_DIM; ++j) {
            W2[i * HIDDEN2_DIM + j] -= learning_rate * (A1[i] * dZ2[j]);
        }
    }
    for (int j = 0; j < HIDDEN2_DIM; ++j) {
        B2[j] -= learning_rate * dZ2[j];
    }

    for (int i = 0; i < INPUT_DIM; ++i) {
        for (int j = 0; j < HIDDEN1_DIM; ++j) {
            W1[i * HIDDEN1_DIM + j] -= learning_rate * (X[i] * dZ1[j]);
        }
    }
    for (int j = 0; j < HIDDEN1_DIM; ++j) {
        B1[j] -= learning_rate * dZ1[j];
    }
#else
    // Error at output layer: dZ2 = A2 - Y
    float dZ2[OUTPUT_DIM];
    for (int i = 0; i < OUTPUT_DIM; i++) {
        dZ2[i] = A2[i] - Y[i];
    }

    // Error at hidden layer: dZ1 = (dZ2 * W2^T) * ReLU_derivative(Z1)
    float dZ1[HIDDEN_DIM] = {0.0f};
    backwardLayer(dZ2, OUTPUT_DIM, Z1, dZ1, HIDDEN_DIM, W2);


    // 3. GRADIENT DESCENT PARAMETER UPDATES
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
#endif
}

