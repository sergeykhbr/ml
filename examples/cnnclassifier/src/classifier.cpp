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
    int InDim = INPUT_DIM;
    for (int i = 0; i < LAYER_NUM; i++) {
        LayerData_[i].prevdim = InDim;
        LayerData_[i].dim = LAYER_DIM[i];
        LayerData_[i].W = new float [InDim * LAYER_DIM[i]];
        LayerData_[i].B = new float [LAYER_DIM[i]];
        LayerData_[i].Z = new float [LAYER_DIM[i]];
        LayerData_[i].A = new float [LAYER_DIM[i]];
        LayerData_[i].dZ = new float [LAYER_DIM[i]];
        LayerData_[i].batchW = new float [InDim * LAYER_DIM[i]];
        LayerData_[i].adamW_M = new float [InDim * LAYER_DIM[i]];
        LayerData_[i].adamW_V = new float [InDim * LAYER_DIM[i]];
        LayerData_[i].batchB = new float [LAYER_DIM[i]];
        LayerData_[i].adamB_M = new float [LAYER_DIM[i]];
        LayerData_[i].adamB_V = new float [LAYER_DIM[i]];

        float scale = std::sqrt(2.0f / InDim);
        std::normal_distribution<float> dist(0.0f, scale);
        for (int ii = 0; ii < InDim * LAYER_DIM[i]; ++ii) {
            LayerData_[i].W[ii] = dist(gen);
            LayerData_[i].batchW[ii] = 0.0f;
            LayerData_[i].adamW_M[ii] = 0.0f;
            LayerData_[i].adamW_V[ii] = 0.0f;
        }
        for (int ii = 0; ii < LAYER_DIM[i]; ++ii) {
            LayerData_[i].B[ii] = 0.0f;
            LayerData_[i].batchB[ii] = 0.0f;
            LayerData_[i].adamB_M[ii] = 0.0f;
            LayerData_[i].adamB_V[ii] = 0.0f;
        }

        InDim = LAYER_DIM[i];
    }
    OutputData_.prevdim = InDim;
    OutputData_.dim = OUTPUT_DIM;
    OutputData_.W = new float [InDim * OUTPUT_DIM];
    OutputData_.B = new float [OUTPUT_DIM];
    OutputData_.Z = new float [OUTPUT_DIM];
    OutputData_.A = new float [OUTPUT_DIM];
    OutputData_.dZ = new float [OUTPUT_DIM];
    OutputData_.batchW = new float [InDim * OUTPUT_DIM];
    OutputData_.adamW_M = new float [InDim * OUTPUT_DIM];
    OutputData_.adamW_V = new float [InDim * OUTPUT_DIM];
    OutputData_.batchB = new float [OUTPUT_DIM];
    OutputData_.adamB_M = new float [OUTPUT_DIM];
    OutputData_.adamB_V = new float [OUTPUT_DIM];

    float scale = std::sqrt(2.0f / InDim);
    std::normal_distribution<float> dist(0.0f, scale);
    for (int ii = 0; ii < InDim * OUTPUT_DIM; ++ii) {
        OutputData_.W[ii] = dist(gen);
        OutputData_.batchW[ii] = 0.0f;
        OutputData_.adamW_M[ii] = 0.0f;
        OutputData_.adamW_V[ii] = 0.0f;
    }
    for (int ii = 0; ii < OUTPUT_DIM; ++ii) {
        OutputData_.B[ii] = 0.0f;
        OutputData_.batchB[ii] = 0.0f;
        OutputData_.adamB_M[ii] = 0.0f;
        OutputData_.adamB_V[ii] = 0.0f;
    }
    batchSize_ = 0;
    batchNum_ = 0;
}

MoonClassifier::~MoonClassifier() {
    for (int i = 0; i < LAYER_NUM; i++) {
        delete [] LayerData_[i].W;
        delete [] LayerData_[i].B;
        delete [] LayerData_[i].Z;
        delete [] LayerData_[i].A;
        delete [] LayerData_[i].dZ;
        delete [] LayerData_[i].batchW;
        delete [] LayerData_[i].adamW_M;
        delete [] LayerData_[i].adamW_V;
        delete [] LayerData_[i].batchB;
        delete [] LayerData_[i].adamB_M;
        delete [] LayerData_[i].adamB_V;
    }
    delete [] OutputData_.W;
    delete [] OutputData_.B;
    delete [] OutputData_.Z;
    delete [] OutputData_.A;
    delete [] OutputData_.dZ;
    delete [] OutputData_.batchW;
    delete [] OutputData_.adamW_M;
    delete [] OutputData_.adamW_V;
    delete [] OutputData_.batchB;
    delete [] OutputData_.adamB_M;
    delete [] OutputData_.adamB_V;
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

void MoonClassifier::backwardLayer(float *dZ, int sz,                   // Layer[n]
                                   float *Zprv, float *dZprv, int prvsz,// Layer[n-1]
                                   float *W) {                          // Layer[n]
    // Error at hidden layer: dOUT = (dIN * W^T) * ReLU_derivative(OUT)
    for (int i = 0; i < prvsz; ++i) {
        float error = 0.0f;
        for (int j = 0; j < sz; ++j) {
            error += dZ[j] * W[i * sz + j]; 
        }
#ifdef SIGMOID_ENA
        dZprv[i] = error * derivativeSigmoid(Zprv[i]);
#else
        dZprv[i] = error * derivativeReLU(Zprv[i]);
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
    LayerDataType *layer;
    int InDim = INPUT_DIM;
    for (int i = 0; i < LAYER_NUM; i++) {
        layer = &LayerData_[i];
        forwardLayer(IN, InDim,
                     layer->Z, LAYER_DIM[i],
                     layer->W,
                     layer->B);
        activation(layer->Z, layer->A, LAYER_DIM[i]);

        IN = layer->A;
        InDim = LAYER_DIM[i];
    }
    forwardLayer(IN, InDim,
                OutputData_.Z, OUTPUT_DIM,
                OutputData_.W,
                OutputData_.B);
    activationSoftmax(OutputData_.Z, OUT, OUTPUT_DIM);

    return OUT[1];
}

void MoonClassifier::gradientDescent(float *dZ, int sz,   // Layer[n]
                                     float *Aprv, int prvsz,     // Layer[n-1]
                                     float *W,              // Layer[n]
                                     float *B) {            // Layer[n]
    for (int i = 0; i < prvsz; ++i) {
        for (int j = 0; j < sz; ++j) {
            W[i * sz + j] += (Aprv[i] * dZ[j]);
        }
    }
    for (int i = 0; i < sz; i++) {
        B[i] += dZ[i];
    }
}

void MoonClassifier::batchIncrement(int batchNum,
                                    int batchSize,
                                    float learning_rate,
                                    LayerDataType *layer) {
    float grad;
#ifdef ADAM_ENA
    float beta1_correction = 1.0f - std::pow(ADAM_BETA1, batchNum);
    float beta2_correction = 1.0f - std::pow(ADAM_BETA2, batchNum);
#endif
    for (int i = 0; i < layer->prevdim * layer->dim; i++) {
        grad = layer->batchW[i] / batchSize;
#ifdef ADAM_ENA
        // Update first moment (direct LP filter)
        layer->adamW_M[i] = ADAM_BETA1 * layer->adamW_M[i]
                        + (1.0f - ADAM_BETA1) * grad;
        // Update second moment (Noise power LP filter)
        layer->adamW_V[i] = ADAM_BETA2 * layer->adamW_V[i]
                        + (1.0f - ADAM_BETA2) * grad * grad;
        // Bias correction
        float m_hat = layer->adamW_M[i] / beta1_correction;
        float v_hat = layer->adamW_V[i] / beta2_correction;

        layer->W[i] -= (ADAM_ALPHA / (std::sqrtf(v_hat) + ADAM_EPSILON)) * m_hat;
#else
        layer->W[i] -= learning_rate * grad;
#endif
    }
    for (int i = 0; i < layer->dim; i++) {
        grad = layer->batchB[i] / batchSize;
#ifdef ADAM_ENA
        // Update first moment (direct LP filter)
        layer->adamB_M[i] = ADAM_BETA1 * layer->adamB_M[i]
                        + (1.0f - ADAM_BETA1) * grad;
        // Update second moment (Noise power LP filter)
        layer->adamB_V[i] = ADAM_BETA2 * layer->adamB_V[i]
                        + (1.0f - ADAM_BETA2) * grad * grad;
        // Bias correction
        float m_hat = layer->adamB_M[i] / beta1_correction;
        float v_hat = layer->adamB_V[i] / beta2_correction;

        layer->B[i] -= (ADAM_ALPHA / (std::sqrtf(v_hat) + ADAM_EPSILON)) * m_hat;
#else
        layer->B[i] -= learning_rate * grad;
#endif
    }
    memset(layer->batchW, 0, layer->prevdim * layer->dim * sizeof(float));
    memset(layer->batchB, 0, layer->dim * sizeof(float));
}

void MoonClassifier::trainStep(DataPoint *datapoint) {
    // --- 1. FORWARD PASS ---
    float X[INPUT_DIM] = {datapoint->x, datapoint->y};

    forwardPass(X, OutputData_.A);
    // Calculate loss for tracking: Cross-Entropy = -sum(Y * log(A3))
    float loss = -std::log(std::max(OutputData_.A[datapoint->label], 1e-7f));

    // Calculate Target Vector Y (One-hot mapping)
    float Y[OUTPUT_DIM] = {0.0f};
    Y[datapoint->label] = 1.0f;


    // 2. BACKWARD PASS
    for (int i = 0; i < OUTPUT_DIM; i++) {
        OutputData_.dZ[i] = OutputData_.A[i] - Y[i];
    }

    LayerDataType *layer = &OutputData_;
    int LayerDim = OUTPUT_DIM;
    for (int i = LAYER_NUM - 1; i >= 0; i--) {
        backwardLayer(layer->dZ, LayerDim,
                      LayerData_[i].Z, LayerData_[i].dZ, LAYER_DIM[i],
                      layer->W);
        gradientDescent(layer->dZ, LayerDim,
                        LayerData_[i].A, LAYER_DIM[i],
                        layer->batchW,
                        layer->batchB);
        layer = &LayerData_[i];
        LayerDim = LAYER_DIM[i];
    }
    gradientDescent(layer->dZ, LayerDim,
                    X, INPUT_DIM,
                    layer->batchW,
                    layer->batchB);


    if (++batchSize_ >= MINI_BATCH_SIZE) {
        batchNum_++;
        batchIncrement(batchNum_,
                       batchSize_,
                       LEARNING_RATE,
                       &OutputData_);
        for (int i = LAYER_NUM - 1; i >= 0; i--) {
            batchIncrement(batchNum_,
                           batchSize_,
                           LEARNING_RATE,
                           &LayerData_[i]);
        }
        batchSize_ = 0;
    }
}

