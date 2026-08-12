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

CNNClassifier::CNNClassifier(std::mt19937 &gen) {
    float scale = std::sqrt(2.0f / KERNEL_DIM);
    std::normal_distribution<float> dist(0.0f, scale);
    LayerConv_.dim = FLATTEN_DIM;
    LayerConv_.prevdim = IMG_H * IMG_W;
    for (int i = 0; i < NUM_FILTERS * KERNEL_DIM; i++) {
        LayerConv_.K[i] = dist(gen);
        LayerConv_.batchK[i] = 0.0f;
        LayerConv_.adamK_M[i] = 0.0f;
        LayerConv_.adamK_V[i] = 0.0f;
    }
    for (int i = 0; i < FLATTEN_DIM; i++) {
        LayerConv_.Z[i] = 0.0f;
        LayerConv_.A[i] = 0.0f;
    }
    for (int i = 0; i < NUM_FILTERS; i++) {
        LayerConv_.B[i] = 0;
        LayerConv_.batchB[i] = 0;
        LayerConv_.adamB_M[i] = 0;
        LayerConv_.adamB_V[i] = 0;
    }

    int InDim = INPUT_DIM;
    /*for (int i = 0; i < LAYER_NUM; i++) {
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
    }*/
    InDim = FLATTEN_DIM;
    {
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

        scale = std::sqrt(2.0f / InDim);
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
    }
    batchSize_ = 0;
    batchNum_ = 0;
}

CNNClassifier::~CNNClassifier() {
    /*for (int i = 0; i < LAYER_NUM; i++) {
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
    }*/
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
void CNNClassifier::forwardLayer(float *IN, int isz,
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

float CNNClassifier::ReLU(float IN) {
    return std::max(0.0f, IN);
}

float CNNClassifier::derivativeReLU(float IN) {
    return IN > 0.0f ? 1.0f : 0.0f;
}

float CNNClassifier::Sigmoid(float IN) {
    return 1.0f / (1.0f + std::exp(-IN));
}

float CNNClassifier::derivativeSigmoid(float IN) {
    float act = Sigmoid(IN);
    return act * (1.0f - act);
}


void CNNClassifier::activation(float *IN, float *OUT, int sz) {
    for (int i = 0; i < sz; ++i) {
#ifdef SIGMOID_ENA
        OUT[i] = Sigmoid(IN[i]);
#else
        OUT[i] = ReLU(IN[i]);
#endif
    }
}

void CNNClassifier::activationSoftmax(float *IN, float *OUT, int sz) {
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

float CNNClassifier::correlateImage(float *img, int x, int y, float *filt) {
    float sum = 0.0f;
    // Slide the 3x3 patch window
    for (int ky = 0; ky < KERNEL_SIZE; ky++) {
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            float pixel = img[(y + ky) * IMG_W + x + kx];
            float weight = filt[ky * KERNEL_SIZE + kx];
            sum += pixel * weight;
        }
    }
    return sum;
}

void CNNClassifier::forwardPass(float *IN, float *OUT) {
    ConvLayerType *layer = &LayerConv_;
    int InDim = INPUT_DIM;

    for (int f = 0; f < NUM_FILTERS; ++f) {
        int feature_map_offset = f * OUT_W * OUT_H;

        for (int out_y = 0; out_y < OUT_H; out_y++) {
            for (int out_x = 0; out_x < OUT_W; out_x++) {
                float sum = correlateImage(IN, out_x, out_y, &layer->K[f * KERNEL_DIM]);
                int out_idx = feature_map_offset + out_y * OUT_W + out_x;
                layer->Z[out_idx] = sum + layer->B[f];
                layer->A[out_idx] = ReLU(layer->Z[out_idx]);
            }
        }
    }

    IN = LayerConv_.A;
    InDim = FLATTEN_DIM;
    forwardLayer(IN, InDim,
                OutputData_.Z, OUTPUT_DIM,
                OutputData_.W,
                OutputData_.B);
    activationSoftmax(OutputData_.Z, OutputData_.A, OUTPUT_DIM);

    /*for (int i = 0; i < LAYER_NUM; i++) {
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
    */
}

void CNNClassifier::backwardLayer(float *dZ, int sz,                   // Layer[n]
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

void CNNClassifier::backwardLayer(LayerDataType *IN, ConvLayerType *OUT) {
    for (int i = 0; i < OUT->dim; ++i) {
        float error = 0.0f;
        for (int j = 0; j < IN->dim; ++j) {
            error += IN->dZ[j] * IN->W[i * IN->dim + j]; 
        }
#ifdef SIGMOID_ENA
        OUT->dZ[i] = error * derivativeSigmoid(OUT->Z[i]);
#else
        OUT->dZ[i] = error * derivativeReLU(OUT->Z[i]);
#endif
    }
}

void CNNClassifier::gradientDescent(float *dZ, int sz,   // Layer[n]
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

void CNNClassifier::gradientDescent(LayerDataType *UP, float *IN) {
    for (int i = 0; i < UP->prevdim; ++i) {
        for (int j = 0; j < UP->dim; ++j) {
            UP->batchW[i * UP->dim + j] += (IN[i] * UP->dZ[j]);
        }
    }
    for (int i = 0; i < UP->dim; i++) {
        UP->batchB[i] += UP->dZ[i];
    }
}

void CNNClassifier::gradientDescent(ConvLayerType *UP, float *BTM) {
    for (int f = 0; f < NUM_FILTERS; ++f) {
        int filter_offset = f * KERNEL_SIZE * KERNEL_SIZE;
        int feature_map_offset = f * OUT_W * OUT_H;

        for (int out_y = 0; out_y < OUT_H; ++out_y) {
            for (int out_x = 0; out_x < OUT_W; ++out_x) {
                int out_idx = feature_map_offset + out_y * OUT_W + out_x;
                float current_dZ1 = UP->dZ[out_idx];
                UP->batchB[f] += current_dZ1;

                // Distribute gradient across the 3x3 filter window locations
                for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                    for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                        int img_x = out_x + kx;
                        int img_y = out_y + ky;
                        float input_pixel = BTM[img_y * IMG_W + img_x];

                        UP->batchK[filter_offset + ky * KERNEL_SIZE + kx] += input_pixel * current_dZ1;
                    }
                }
            }
        }
    }
}

void CNNClassifier::batchIncrement(int batchNum,
                                    int batchSize,
                                    float learning_rate,
                                    LayerDataType *layer) {
    float grad;
    float beta1_correction = 1.0f - std::pow(ADAM_BETA1, batchNum);
    float beta2_correction = 1.0f - std::pow(ADAM_BETA2, batchNum);
    for (int i = 0; i < layer->prevdim * layer->dim; i++) {
        grad = layer->batchW[i] / batchSize;
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
    }
    for (int i = 0; i < layer->dim; i++) {
        grad = layer->batchB[i] / batchSize;
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
    }
    memset(layer->batchW, 0, layer->prevdim * layer->dim * sizeof(float));
    memset(layer->batchB, 0, layer->dim * sizeof(float));
}

void CNNClassifier::batchIncrement(int batchNum,
                                    int batchSize,
                                    float learning_rate,
                                    ConvLayerType *layer) {
    float grad;
    float beta1_correction = 1.0f - std::pow(ADAM_BETA1, batchNum);
    float beta2_correction = 1.0f - std::pow(ADAM_BETA2, batchNum);
    for (int i = 0; i < NUM_FILTERS * KERNEL_DIM; i++) {
        grad = layer->batchK[i] / batchSize;
#if 0
        layer->K[i] -= learning_rate * grad;
#else
        // Update first moment (direct LP filter)
        layer->adamK_M[i] = ADAM_BETA1 * layer->adamK_M[i]
                        + (1.0f - ADAM_BETA1) * grad;
        // Update second moment (Noise power LP filter)
        layer->adamK_V[i] = ADAM_BETA2 * layer->adamK_V[i]
                        + (1.0f - ADAM_BETA2) * grad * grad;
        // Bias correction
        float m_hat = layer->adamK_M[i] / beta1_correction;
        float v_hat = layer->adamK_V[i] / beta2_correction;

        layer->K[i] -= (ADAM_ALPHA / (std::sqrtf(v_hat) + ADAM_EPSILON)) * m_hat;
#endif
    }
    for (int i = 0; i < NUM_FILTERS; i++) {
        grad = layer->batchB[i] / batchSize;
#if 0
        layer->B[i] -= learning_rate * grad;
#else
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
#endif
    }
    memset(layer->batchK, 0, NUM_FILTERS * KERNEL_DIM * sizeof(float));
    memset(layer->batchB, 0, NUM_FILTERS * sizeof(float));
}

void CNNClassifier::trainStep(DataPoint *datapoint) {
    float *X = datapoint->pixels;   // input layer

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

    backwardLayer(&OutputData_, &LayerConv_);
    gradientDescent(&OutputData_, LayerConv_.A);

    gradientDescent(&LayerConv_, X);

    /*LayerDataType *layer = &OutputData_;
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

    */
    if (++batchSize_ >= MINI_BATCH_SIZE) {
        batchNum_++;
        batchIncrement(batchNum_,
                       batchSize_,
                       LEARNING_RATE,
                       &OutputData_);
        /*for (int i = LAYER_NUM - 1; i >= 0; i--) {
            batchIncrement(batchNum_,
                           batchSize_,
                           LEARNING_RATE,
                           &LayerData_[i]);
        }*/
            batchIncrement(batchNum_,
                           batchSize_,
                           LEARNING_RATE,
                           &LayerConv_);
        batchSize_ = 0;
    }
}

