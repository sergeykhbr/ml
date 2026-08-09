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

#include <cmath>

// Input parameters:
const int DATA_SET_SIZE = 1024;
const int MINI_BATCH_SIZE = 32;
const int TRAIN_EPOCH_TOTAL = 20;
const float IN_DATA_DISTRIBUTION = 0.2f;
const float LEARNING_RATE = 0.05f * std::sqrtf(MINI_BATCH_SIZE);

#define FIXED_SEED 0x11223344
//#define SIGMOID_ENA

// Architecture Parameters
const int INPUT_DIM = 2;   // (x, y) coordinates
const int OUTPUT_DIM = 2;  // 2 classes: Red (0) or Blue (1)
const int LAYER_DIM[] = {24, 14};
const int LAYER_NUM = sizeof(LAYER_DIM)/sizeof(int);    // hidden layers count (min = 1)

struct DataPoint {
    float x;
    float y;
    int label; // INPUT_DIM: 0 for Red, 1 for Blue
};

// Model Parameters (Weights and Biases flattened into 1D memory)
struct LayerDataType {
    int prevdim;    // Previous layer dimension
    int dim;        // Current layer dimension
    float *W;       // Weight
    float *batchW;  // Delta Weight per batch
    float *B;       // Bias
    float *batchB;  // Delta B per batch
    float *Z;       // Hidden Layer: Z = IN * W + B
    float *A;       // Activation: A = ReLU(Z)
    float *dZ;      // backward propogation gradient
};
