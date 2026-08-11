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
#define MINI_BATCH_SIZE 1

const int TRAIN_EPOCH_TOTAL = 50;

// Input parameters:
const int DATA_SET_SIZE = 1024;
const float LEARNING_RATE = 0.05f;
// ADAM method
const float ADAM_ALPHA = LEARNING_RATE;
const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.999f;
const float ADAM_EPSILON = 1e-8f;


// Architecture Parameters
const int IMG_W = 14;
const int IMG_H = 14;
const int KERNEL_SIZE = 3; // 3x3 pixel filter to detect edges
const int KERNEL_DIM = KERNEL_SIZE * KERNEL_SIZE;
const int NUM_FILTERS = 3;

const int INPUT_PIXELS = IMG_W * IMG_H;
const int OUT_W = IMG_W - KERNEL_SIZE + 1;
const int OUT_H = IMG_H - KERNEL_SIZE + 1;

const int INPUT_DIM = IMG_W * IMG_H;    // 14*14 = 196 neurons
const int FLATTEN_DIM = NUM_FILTERS * OUT_W * OUT_H;
const int LAYER_DIM[] = {FLATTEN_DIM};  // 3*12*12 = 432 neurons
const int LAYER_NUM = sizeof(LAYER_DIM)/sizeof(int);    // hidden layers count (min = 1)
const int OUTPUT_DIM = 3;  // 3 classes: 0=Circle, 1=Square; 2=Triangle

struct DataPoint {
    float pixels[INPUT_PIXELS];
    int label; // 0=Circle, 1=Square; 2=Triangle
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

struct ConvLayerType {
    int prevdim;    // Previous layer dimension
    int dim;        // Current layer dimension
    float K[NUM_FILTERS * KERNEL_DIM];
    float batchK[NUM_FILTERS * KERNEL_DIM];
    float B[NUM_FILTERS];
    float batchB[NUM_FILTERS];
    float Z[FLATTEN_DIM];       // 
    float A[FLATTEN_DIM];       // Activation: A = ReLU(Z)
    float dZ[FLATTEN_DIM];      // backward propogation gradient
};
