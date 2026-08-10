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
    for (int r = 0; r < IMG_DIM; ++r) {
        for (int c = 0; c < IMG_DIM; ++c) {
            R = std::sqrt((r - center_r) * (r - center_r) + (c - center_c) * (c - center_c));
            if (std::abs(R - radius) < 1.0f || R < radius) {
                sample.pixels[r * IMG_DIM + c] = 0.9f + dist(gen);
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

    for (int r = 3; r < (IMG_DIM - 3); ++r) {
        for (int c = 3; c < (IMG_DIM - 3); ++c) {
            sample.pixels[r * IMG_DIM + c] = 0.9f + dist(gen);
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

    for (int r = 3; r < (IMG_DIM - 3); ++r) {
        int width = r - 3; 
        int start_c = 6 - (width / 2);
        int end_c = 6 + (width / 2);
        for (int c = start_c; c <= end_c; ++c) {
            if (c >= 0 && c < IMG_DIM) {
                sample.pixels[r * IMG_DIM + c] = 0.9f + dist(gen);
            }
        }
    }

    for (int i = 0; i < INPUT_PIXELS; ++i) {
        sample.pixels[i] = std::max(0.0f, std::min(1.0f, sample.pixels[i]));
    }
}

void generate_data_set(std::mt19937 &gen,
                       std::vector<DataPoint> &dataset,
                       int sample_per_class) {
    // Noise distribution to blend the spiral arms slightly
    float spiral_noise_bound = 0.03f;
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    DataPoint sample;

    for (int i = 0; i < sample_per_class; ++i) {
        generate_circle(gen, sample);
        dataset.push_back(sample);
 
        generate_square(gen, sample);
        dataset.push_back(sample);

        generate_triangle(gen, sample);
        dataset.push_back(sample);
    }
}


