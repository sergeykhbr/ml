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

    PlotWidget plot(gen, seed);
    plot.setWindowTitle("CNN");
    plot.resize(480, 240);

    plot.show();
    app.exec();

    return 0;
}
