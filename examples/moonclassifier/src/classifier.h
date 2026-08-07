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
    float forwardPass(float *IN, float *OUT);

 private:
    // Model Parameters (Weights and Biases flattened into 1D memory)
    float W1[INPUT_DIM * HIDDEN_DIM];
    float B1[HIDDEN_DIM];
    float Z1[HIDDEN_DIM];   // Hidden Layer: Z1 = IN * W1 + B1
    float A1[HIDDEN_DIM];   // Activation: A1 = ReLU(Z1)

    float W2[HIDDEN_DIM * OUTPUT_DIM];
    float B2[OUTPUT_DIM];
    float Z2[OUTPUT_DIM];   // Output Layer: Z2 = A1 * W2 + B2

    float learning_rate = 0.05f;

};
