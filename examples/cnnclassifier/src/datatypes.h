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

#define FIXED_SEED 0x11223344
//#define SIGMOID_ENA
#define ADAM_ENA        // if disable Mini Batch used (simple average)
#define MINI_BATCH_SIZE 32

const int TRAIN_EPOCH_TOTAL = 50;

// Input parameters:
const int DATA_SET_SIZE = 1024;
#if !defined(ADAM_ENA) && (MINI_BATCH_SIZE > 1)
    const float LEARNING_RATE = 0.05f * std::sqrtf(MINI_BATCH_SIZE);
#else
    const float LEARNING_RATE = 0.05f;
#endif
// ADAM method
const float ADAM_ALPHA = LEARNING_RATE;
const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.999f;
const float ADAM_EPSILON = 1e-8f;


// Architecture Parameters
const int IMG_DIM = 14;
const int INPUT_PIXELS = IMG_DIM * IMG_DIM;

const int KERNEL_DIM = 3; // 3x3 pixel filter to detect edges

const int INPUT_DIM = IMG_DIM * IMG_DIM;
const int OUTPUT_DIM = 3;  // 3 classes: 0=Circle, 1=Square; 2=Triangle
const int LAYER_DIM[] = {144};  //(IMG_DIM - KERNEL_DIM + 1) * (IMG_DIM - KERNEL_DIM + 1);
const int LAYER_NUM = sizeof(LAYER_DIM)/sizeof(int);    // hidden layers count (min = 1)

struct DataPoint {
    float pixels[INPUT_PIXELS];
    int label; // INPUT_DIM: 0=Circle, 1=Square; 2=Triangle
};

// Model Parameters (Weights and Biases flattened into 1D memory)
struct LayerDataType {
    int prevdim;    // Previous layer dimension
    int dim;        // Current layer dimension
    float *W;       // Weight
    float *B;       // Bias
    float *Z;       // Hidden Layer: Z = IN * W + B
    float *A;       // Activation: A = ReLU(Z)
    float *dZ;      // backward propogation gradient
    // Batch
    float *batchW;  // Delta Weight per batch
    float *adamW_M; // First moment (Direct LP filter)
    float *adamW_V; // Second moment (Noise power LP filter)
    float *batchB;  // Delta B per batch
    float *adamB_M; // First moment (Direct LP filter)
    float *adamB_V; // Second moment (Noise power LP filter)
};
