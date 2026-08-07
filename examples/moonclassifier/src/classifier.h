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

#pragma once

#include <datatypes.h>
#include <random>

class MoonClassifier {
 public:
    explicit MoonClassifier(std::mt19937 &gen);

    void trainStep(DataPoint *datapoint);
    void forwardLayer(float *IN, int isz, float *OUT, int osz, float *W, float *B);
    void activation(float *IN, float *OUT, int sz);
    void activationSoftmax(float *IN, float *OUT, int sz);
    float forwardPass(float *IN, float *OUT);
    void backwardLayer(float *dIN, int isz, float *OUT, float *dOUT, int osz, float *W);

 private:
    // Model Parameters (Weights and Biases flattened into 1D memory)
#ifdef LAYER2_ENA
    float W1[INPUT_DIM * HIDDEN1_DIM];
    float B1[HIDDEN1_DIM];
    float Z1[HIDDEN1_DIM];   // Hidden Layer: Z1 = IN * W1 + B1
    float A1[HIDDEN1_DIM];   // Activation: A1 = ReLU(Z1)

    float W2[HIDDEN1_DIM * HIDDEN2_DIM];
    float B2[HIDDEN2_DIM];
    float Z2[HIDDEN2_DIM];   // Hidden Layer: Z2 = Z1 * W2 + B2
    float A2[HIDDEN2_DIM];   // Activation: A2 = ReLU(Z2)

    float W3[HIDDEN2_DIM * OUTPUT_DIM];
    float B3[OUTPUT_DIM];
    float Z3[OUTPUT_DIM];   // Output Layer: Z3 = A2 * W3 + B3
#else
    float W1[INPUT_DIM * HIDDEN_DIM];
    float B1[HIDDEN_DIM];
    float Z1[HIDDEN_DIM];   // Hidden Layer: Z1 = IN * W1 + B1
    float A1[HIDDEN_DIM];   // Activation: A1 = ReLU(Z1)

    float W2[HIDDEN_DIM * OUTPUT_DIM];
    float B2[OUTPUT_DIM];
    float Z2[OUTPUT_DIM];   // Output Layer: Z2 = A1 * W2 + B2
#endif

    float learning_rate = 0.05f;

};
