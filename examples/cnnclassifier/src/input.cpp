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

#include "datatypes.h"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

void generate_circle(std::mt19937 &gen,
                     DataPoint &sample) {

    float center_r = 6.5f;
    float center_c = 6.5f;
    float radius = 3.5f;
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    sample.label = 0;   // circle
    // Fill canvas with base gray background + noise
    for (int i = 0; i < INPUT_PIXELS; ++i) {
        sample.pixels[i] = 0.1f + dist(gen);
    }

    float R;
    for (int r = 0; r < IMG_W; ++r) {
        for (int c = 0; c < IMG_H; ++c) {
            R = std::sqrt((r - center_r) * (r - center_r) + (c - center_c) * (c - center_c));
            if (std::abs(R - radius) < 1.0f || R < radius) {
                sample.pixels[r * IMG_W + c] = 0.9f + dist(gen);
            }
        }
    }

    for (int i = 0; i < INPUT_PIXELS; ++i) {
        sample.pixels[i] = std::max(0.0f, std::min(1.0f, sample.pixels[i]));
    }
}

void generate_square(std::mt19937 &gen,
                     DataPoint &sample) {

    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    sample.label = 1;   // square
    // Fill canvas with base gray background + noise
    for (int i = 0; i < INPUT_PIXELS; ++i) {
        sample.pixels[i] = 0.1f + dist(gen);
    }

    for (int r = KERNEL_SIZE; r < (IMG_W - KERNEL_SIZE); ++r) {
        for (int c = KERNEL_SIZE; c < (IMG_H - KERNEL_SIZE); ++c) {
            sample.pixels[r * IMG_W + c] = 0.9f + dist(gen);
        }
    }

    for (int i = 0; i < INPUT_PIXELS; ++i) {
        sample.pixels[i] = std::max(0.0f, std::min(1.0f, sample.pixels[i]));
    }
}

void generate_triangle(std::mt19937 &gen,
                       DataPoint &sample) {

    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    sample.label = 2;   // triangle
    // Fill canvas with base gray background + noise
    for (int i = 0; i < INPUT_PIXELS; ++i) {
        sample.pixels[i] = 0.1f + dist(gen);
    }

    for (int r = KERNEL_SIZE; r < (IMG_W - KERNEL_SIZE); ++r) {
        int width = r - KERNEL_SIZE; 
        int start_c = 2*KERNEL_SIZE - (width / 2);
        int end_c = 2*KERNEL_SIZE + (width / 2);
        for (int c = start_c; c <= end_c; ++c) {
            if (c >= 0 && c < IMG_H) {
                sample.pixels[r * IMG_W + c] = 0.9f + dist(gen);
            }
        }
    }

    for (int i = 0; i < INPUT_PIXELS; ++i) {
        sample.pixels[i] = std::max(0.0f, std::min(1.0f, sample.pixels[i]));
    }
}

void generate_data(std::mt19937 &gen,
                       DataPoint *data) {
    std::uniform_int_distribution<int> idist(0, 2);
    int type = idist(gen);

    switch(type) {
    case 0:
        generate_circle(gen, *data);
        break;
    case 1:
        generate_square(gen, *data);
        break;
    default:
        generate_triangle(gen, *data);
    }
}


