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

class CNNClassifier {
 public:
    explicit CNNClassifier(std::mt19937 &gen);
    ~CNNClassifier();

    float correlateImage(float *img, int x, int y, float *filt);
    void trainStep(DataPoint *datapoint);
    void forwardLayer(float *IN, int isz, float *OUT, int osz, float *W, float *B);
    float ReLU(float IN);
    float derivativeReLU(float IN);
    float Sigmoid(float IN);
    float derivativeSigmoid(float IN);
    void activation(float *IN, float *OUT, int sz);
    void activationSoftmax(float *IN, float *OUT, int sz);
    void forwardPass(float *IN, float *OUT);
    void backwardLayer(float *dIN, int isz, float *OUT, float *dOUT, int osz, float *W);
    void backwardLayer(LayerDataType *IN, ConvLayerType *OUT);
    void gradientDescent(float *dZ, int dZsz,
                         float *A, int Asz,
                         float *W,
                         float *B);
    void gradientDescent(LayerDataType *TOP, float *IN);
    void gradientDescent(ConvLayerType *TOP, float *IN);
    void batchIncrement(int batchNum,
                        int batchSize,
                        float learning_rate,
                        LayerDataType *layer);
    void batchIncrement(int batchNum,
                        int batchSize,
                        float learning_rate,
                        ConvLayerType *layer);

 private:
    ConvLayerType LayerConv_;
    //LayerDataType LayerData_[LAYER_NUM];
    LayerDataType OutputData_;

    int batchSize_;   // counter from 0 to Batches Total - 1
    int batchNum_;  // number of batches per whole training
};

#if 0
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

// Network Structural Settings
const int IMG_W = 14;
const int IMG_H = 14;
const int INPUT_DIM = IMG_W * IMG_H; // 196 pixels
const int KERNEL_SIZE = 3;
const int NUM_FILTERS = 3;

// Output spatial math
const int OUT_W = IMG_W - KERNEL_SIZE + 1; // 12
const int OUT_H = IMG_H - KERNEL_SIZE + 1; // 12
const int FLATTEN_DIM = NUM_FILTERS * OUT_W * OUT_H; // 3 * 12 * 12 = 432
const int OUTPUT_DIM = 3; // 0: Circle, 1: Square, 2: Triangle

// Simple data structure for single images
struct ImageSample {
    float pixels[INPUT_DIM]; // Normalized values [0.0, 1.0]
    int label;               // 0, 1, or 2
};

class ShapeCNN {
private:
    // Model Parameters (Flattened into 1D memory blocks)
    float K[NUM_FILTERS * KERNEL_SIZE * KERNEL_SIZE]; // Conv Kernels (3 * 3 * 3 = 27)
    float conv_bias[NUM_FILTERS];                    // Conv Biases (3)
    
    float W_dense[FLATTEN_DIM * OUTPUT_DIM];          // Dense Weights (432 * 3 = 1296)
    float B_dense[OUTPUT_DIM];                        // Dense Biases (3)

    float learning_rate = 0.01f; // Standard online single-sample learning rate

public:
    ShapeCNN();

    float trainStep(const float* input_pixels, int target_label, bool print_debug);
    int predict(const float* input_pixels) const;
};

ShapeCNN::ShapeCNN() {
    std::random_device rd;
    std::mt19937 gen(rd());
        
    // Initialize Convolutional Kernels (He initialization)
    float scale_conv = std::sqrt(2.0f / (KERNEL_SIZE * KERNEL_SIZE));
    std::normal_distribution<float> dist_conv(0.0f, scale_conv);
    for (int i = 0; i < NUM_FILTERS * KERNEL_SIZE * KERNEL_SIZE; ++i) K[i] = dist_conv(gen);
    for (int i = 0; i < NUM_FILTERS; ++i) conv_bias[i] = 0.0f;

    // Initialize Dense Weights (He initialization)
    float scale_dense = std::sqrt(2.0f / FLATTEN_DIM);
    std::normal_distribution<float> dist_dense(0.0f, scale_dense);
    for (int i = 0; i < FLATTEN_DIM * OUTPUT_DIM; ++i) W_dense[i] = dist_dense(gen);
    for (int i = 0; i < OUTPUT_DIM; ++i) B_dense[i] = 0.0f;
}

float ShapeCNN::trainStep(const float* input_pixels, int target_label, bool print_debug) {
    // ==========================================
    // --- 1. FORWARD PASS ---
    // ==========================================
    
    // Allocate memory trackers for the feature maps
    float conv_out[FLATTEN_DIM] = {0.0f};
    float conv_act[FLATTEN_DIM] = {0.0f}; // This is the flattened dense layer input (A1)

    // Slide filters over the 2D pixel space
    for (int f = 0; f < NUM_FILTERS; ++f) {
        int filter_offset = f * KERNEL_SIZE * KERNEL_SIZE;
        int feature_map_offset = f * OUT_W * OUT_H;

        for (int out_y = 0; out_y < OUT_H; ++out_y) {
            for (int out_x = 0; out_x < OUT_W; ++out_x) {
                
                float sum = 0.0f;
                // Compute 3x3 patch dot product
                for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                    for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                        int img_x = out_x + kx;
                        int img_y = out_y + ky;
                        
                        float pixel = input_pixels[img_y * IMG_W + img_x];
                        float weight = K[filter_offset + ky * KERNEL_SIZE + kx];
                        sum += pixel * weight;
                    }
                }
                
                int out_idx = feature_map_offset + out_y * OUT_W + out_x;
                conv_out[out_idx] = sum + conv_bias[f];
                conv_act[out_idx] = std::max(0.0f, conv_out[out_idx]); // ReLU(Z1)
            }
        }
    }

    // Dense Layer calculation: Z2 = A1 * W_dense + B_dense
    float Z2[OUTPUT_DIM] = {0.0f};
    for (int j = 0; j < OUTPUT_DIM; ++j) {
        for (int i = 0; i < FLATTEN_DIM; ++i) {
            Z2[j] += conv_act[i] * W_dense[i * OUTPUT_DIM + j];
        }
        Z2[j] += B_dense[j];
    }

    // Output Activation: A2 = Softmax(Z2)
    float A2[OUTPUT_DIM];
    float max_z = Z2[0];
    for (int c = 1; c < OUTPUT_DIM; ++c) if (Z2[c] > max_z) max_z = Z2[c];

    float sum_exp = 0.0f;
    for (int c = 0; c < OUTPUT_DIM; ++c) sum_exp += std::exp(Z2[c] - max_z);
    for (int c = 0; c < OUTPUT_DIM; ++c) A2[c] = std::exp(Z2[c] - max_z) / sum_exp;

    // Loss tracking
    float loss = -std::log(std::max(A2[target_label], 1e-7f));

    // ==========================================
    // --- 2. BACKWARD PASS (THE CALCULUS) ---
    // ==========================================

    // Error at Output Layer: dZ2 = A2 - Y
    float dZ2[OUTPUT_DIM];
    for (int c = 0; c < OUTPUT_DIM; ++c) {
        float Y_c = (target_label == c) ? 1.0f : 0.0f;
        dZ2[c] = A2[c] - Y_c;
    }

    // Backpropagate error into the Flattened array: dFlatten = dZ2 * W_dense^T
    float dFlatten[FLATTEN_DIM] = {0.0f};
    for (int i = 0; i < FLATTEN_DIM; ++i) {
        for (int j = 0; j < OUTPUT_DIM; ++j) {
            dFlatten[i] += dZ2[j] * W_dense[i * OUTPUT_DIM + j];
        }
    }

    // Backpropagate into Conv Activation: dZ1 = dFlatten * ReLU_derivative(Z1)
    float dZ1[FLATTEN_DIM] = {0.0f};
    for (int i = 0; i < FLATTEN_DIM; ++i) {
        dZ1[i] = (conv_out[i] > 0.0f) ? dFlatten[i] : 0.0f;
    }

    // ==========================================
    // --- 3. ONLINE SGD PARAMETER UPDATES ---
    // ==========================================

    // Update Dense Layer Weights and Biases
    for (int i = 0; i < FLATTEN_DIM; ++i) {
        for (int j = 0; j < OUTPUT_DIM; ++j) {
            W_dense[i * OUTPUT_DIM + j] -= learning_rate * (conv_act[i] * dZ2[j]);
        }
    }
    for (int j = 0; j < OUTPUT_DIM; ++j) B_dense[j] -= learning_rate * dZ2[j];

    // Update Convolution Layer Kernels and Biases
    for (int f = 0; f < NUM_FILTERS; ++f) {
        int filter_offset = f * KERNEL_SIZE * KERNEL_SIZE;
        int feature_map_offset = f * OUT_W * OUT_H;

        for (int out_y = 0; out_y < OUT_H; ++out_y) {
            for (int out_x = 0; out_x < OUT_W; ++out_x) {
                int out_idx = feature_map_offset + out_y * OUT_W + out_x;
                float current_dZ1 = dZ1[out_idx];

                // Distribute gradient across the 3x3 filter window locations
                for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                    for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                        int img_x = out_x + kx;
                        int img_y = out_y + ky;
                        float input_pixel = input_pixels[img_y * IMG_W + img_x];

                        K[filter_offset + ky * KERNEL_SIZE + kx] -= learning_rate * (input_pixel * current_dZ1);
                    }
                }
                conv_bias[f] -= learning_rate * current_dZ1;
            }
        }
    }

    if (print_debug) {
        std::cout << "Loss: " << std::fixed << std::setprecision(4) << loss 
                  << " | Predictions -> Circle: " << A2[0]*100 << "%"
                  << " Square: " << A2[1]*100 << "%"
                  << " Triangle: " << A2[2]*100 << "%" << std::endl;
    }

    return loss;
}

int ShapeCNN::predict(const float* input_pixels) const {
    float conv_out[FLATTEN_DIM] = {0.0f};
    for (int f = 0; f < NUM_FILTERS; ++f) {
        int filter_offset = f * KERNEL_SIZE * KERNEL_SIZE;
        int feature_map_offset = f * OUT_W * OUT_H;

        for (int out_y = 0; out_y < OUT_H; ++out_y) {
            for (int out_x = 0; out_x < OUT_W; ++out_x) {
                float sum = 0.0f;
                for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                    for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                        sum += input_pixels[(out_y + ky) * IMG_W + (out_x + kx)] * K[filter_offset + ky * KERNEL_SIZE + kx];
                    }
                }
                int out_idx = feature_map_offset + out_y * OUT_W + out_x;
                conv_out[out_idx] = std::max(0.0f, sum + conv_bias[f]);
            }
        }
    }

    float Z2[OUTPUT_DIM] = {0.0f};
    for (int j = 0; j < OUTPUT_DIM; ++j) {
        for (int i = 0; i < FLATTEN_DIM; ++i) Z2[j] += conv_out[i] * W_dense[i * OUTPUT_DIM + j];
        Z2[j] += B_dense[j];
    }

    int best_class = 0;
    float max_score = Z2[0];
    for (int c = 1; c < OUTPUT_DIM; ++c) {
        if (Z2[c] > max_score) {
            max_score = Z2[c];
            best_class = c;
        }
    }
    return best_class;
}

int main() {
    ShapeCNN model;
    
    // Create one dummy 14x14 sample representing a perfect Triangle (Label 2)
    // In your actual app, populate these pixels using standard shapes or data loaders.
    ImageSample dummy_triangle;
    dummy_triangle.label = 2; 
    for(int i = 0; i < INPUT_DIM; ++i) dummy_triangle.pixels[i] = 0.0f;
    
    // Simulate a basic triangle shape in the pixel grid
    for(int y = 3; y < 11; ++y) {
        for(int x = 7 - (y-3); x <= 7 + (y-3); ++x) {
            if(x >= 0 && x < IMG_W && y >= 0 && y < IMG_H) {
                dummy_triangle.pixels[y * IMG_W + x] = 1.0f;
            }
        }
    }

    std::cout << "Testing Single-Sample Online SGD TrainStep on Dummy Triangle:" << std::endl;
    for (int step = 1; step <= 5; ++step) {
        std::cout << "Step " << step << " -> ";
        model.trainStep(dummy_triangle.pixels, dummy_triangle.label, true);
    }

    return 0;
}

#endif