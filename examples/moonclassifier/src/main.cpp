#include "datatypes.h"
#include "classifier.h"
#include "gui/PlotWidget.h"
#include <QApplication>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

void generate_data_set(std::mt19937 &gen,
                       std::vector<DataPoint> &dataset) {
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
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
    bool st = true;
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    std::random_device rd;              // generate random number from hardware
    std::mt19937 gen(rd());             // start high-quility Mersenne Twister math engine (range 32-bits uint32_t)
    std::vector<DataPoint> dataset;
    MoonClassifier *model = new MoonClassifier();

    generate_data_set(gen, dataset);

    std::cout << "--- Starting Training Optimization ---" << std::endl;
    // Run training over 20 epochs
    for (int epoch = 1; epoch <= 20; ++epoch) {
        // Shuffle the tracking points each epoch to maintain optimization stability
        std::shuffle(dataset.begin(), dataset.end(), gen);
        
        std::cout << "\n--- Epoch " << epoch << " ---" << std::endl;
        // Print the math state for the first point of the epoch to track progression
        model->trainStep(dataset[0].x, dataset[0].y, dataset[0].label, true);

        // Train silently on the rest of the points
        for (size_t i = 1; i < dataset.size(); ++i) {
            model->trainStep(dataset[i].x, dataset[i].y, dataset[i].label, false);
        }
    }

    PlotWidget plot(dataset, model);
    plot.setWindowTitle("moonclsr");
    plot.resize(320, 240);
    plot.show();
    app.exec();
    delete model;

    return 0;
}
