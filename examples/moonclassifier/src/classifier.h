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
    ~MoonClassifier();

    void trainStep(DataPoint *datapoint);
    void forwardLayer(float *IN, int isz, float *OUT, int osz, float *W, float *B);
    float ReLU(float IN);
    float derivativeReLU(float IN);
    float Sigmoid(float IN);
    float derivativeSigmoid(float IN);
    void activation(float *IN, float *OUT, int sz);
    void activationSoftmax(float *IN, float *OUT, int sz);
    float forwardPass(float *IN, float *OUT);
    void backwardLayer(float *dIN, int isz, float *OUT, float *dOUT, int osz, float *W);
    void gradientDescent(float *dIN, int isz, float *OUT, int osz, float *W, float *B);

 private:
    // Model Parameters (Weights and Biases flattened into 1D memory)
    struct LayerDataType {
        float *W;   // Weight
        float *B;   // Bias
        float *Z;   // Hidden Layer: Z = IN * W + B
        float *A;   // Activation: A = ReLU(Z)
        float *dZ;  // backward propogation gradient
    };
    LayerDataType LayerData_[LAYER_NUM];
    LayerDataType OutputData_;

    float learning_rate = 0.05f;

};
