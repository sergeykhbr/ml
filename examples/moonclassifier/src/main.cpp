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
#include "classifier.h"
#include "gui/PlotWidget.h"
#include <QApplication>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

void generate_spiral_data_set(std::mt19937 &gen,
                              std::vector<DataPoint> &dataset) {
    // Noise distribution to blend the spiral arms slightly
    float spiral_noise_bound = 0.03f;
    std::uniform_real_distribution<float> dist(-spiral_noise_bound, spiral_noise_bound);
    DataPoint pmin = {0,0,0};
    DataPoint pmax = {0,0,0};
    
    const int points_per_arm = 400; // 400 * 3 arms = 1200 total dataset points
    const float max_radius = 1.0f;  // How far out the spiral stretches
    const float turns = 1.3f;       // Number of full rotations per arm

    for (int arm = 0; arm < 3; ++arm) {
        // Phase shift: Arm 0 = 0, Arm 1 = 120 degrees, Arm 2 = 240 degrees
        float phase_offset = arm * (2.0f * 3.14159265f / 3.0f);

        for (int i = 0; i < points_per_arm; ++i) {
            // Normalize current step progress (0.0 to 1.0)
            float t = static_cast<float>(i) / points_per_arm;

            // Math: Radius expands linearly, Angle increases over turns
            float radius = t * max_radius;
            float theta = t * (turns * 2.0f * 3.14159265f) + phase_offset;

            // Base clean coordinates
            float x = radius * std::cos(theta);
            float y = radius * std::sin(theta);

            // Inject dataset noise
            x += dist(gen);
            y += dist(gen);

            // Add to dataset with labels: 0, 1, or 2
            if (i == 0) {
                pmin.x = x;
                pmin.y = y;
                pmax.x = x;
                pmax.y = y;
            } else {
                if (x < pmin.x) {
                    pmin.x = x;
                } else if (x > pmax.x) {
                    pmax.x = x;
                }
                if (y < pmin.y) {
                    pmin.y = y;
                } else if (y > pmax.y) {
                    pmax.y = y;
                }
            }
            dataset.push_back({x, y, arm});
        }
    }
}


void generate_data_set(std::mt19937 &gen,
                       std::vector<DataPoint> &dataset) {
    std::uniform_real_distribution<float> dist(-IN_DATA_DISTRIBUTION, IN_DATA_DISTRIBUTION);
    DataPoint pmin = {0,0,0};
    DataPoint pmax = {0,0,0};

    // Create Red Moon coordinates along a curve
    float x;
    float y;
    for (int i = 0; i < 500; ++i) {
        float theta = (i / 500.0f) * 3.14159f;
        x = std::cos(theta) + dist(gen);
        y = std::sin(theta) + dist(gen);
        if (i == 0) {
            pmin.x = x;
            pmin.y = y;
            pmax.x = x;
            pmax.y = y;
        } else {
            if (x < pmin.x) {
                pmin.x = x;
            } else if (x > pmax.x) {
                pmax.x = x;
            }
            if (y < pmin.y) {
                pmin.y = y;
            } else if (y > pmax.y) {
                pmax.y = y;
            }
        }
        dataset.push_back({x, y, 0});
    }
    // Create Blue Moon coordinates shifted downward/sideways
    for (int i = 0; i < 500; ++i) {
        float theta = (i / 500.0f) * 3.14159f;
        x = 1.0f - std::cos(theta) + dist(gen);
        y = 0.5f - std::sin(theta) + dist(gen);
        dataset.push_back({x, y, 1});
    }
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);       

    uint32_t seed;
#ifdef FIXED_SEED
    seed = FIXED_SEED;
    std::mt19937 gen(seed);
#else
    std::random_device rd;              // generate random number from hardware
    seed = rd();
    std::mt19937 gen(rd());             // start high-quility Mersenne Twister math engine (range 32-bits uint32_t)
#endif
    std::vector<DataPoint> dataset;
#ifdef SPIRAL_ENA
    generate_spiral_data_set(gen, dataset);
#else
    generate_data_set(gen, dataset);
#endif

    MoonClassifier *model = new MoonClassifier(gen);
    PlotWidget plot(dataset, model, seed);
    plot.setWindowTitle("moonclsr");
    plot.resize(320, 240);

    std::cout << "--- Starting Training Optimization ---" << std::endl;
    // Run training over 20 epochs
    for (int epoch = 1; epoch <= TRAIN_EPOCH_TOTAL; ++epoch) {
        // Shuffle the tracking points each epoch to maintain optimization stability
        std::shuffle(dataset.begin(), dataset.end(), gen);
        
        std::cout << "\n--- Epoch " << epoch << " ---" << std::endl;
        // Print the math state for the first point of the epoch to track progression
        model->trainStep(&dataset[0]);

        // Train silently on the rest of the points
        for (size_t i = 1; i < dataset.size(); ++i) {
            model->trainStep(&dataset[i]);
        }

        plot.saveEpoch(epoch);
    }

    plot.show();
    app.exec();
    delete model;

    return 0;
}
