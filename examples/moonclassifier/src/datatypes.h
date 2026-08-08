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
// Input parameters:
const float IN_DATA_DISTRIBUTION = 0.2f;

//#define LAYER2_ENA
// Architecture Parameters
const int INPUT_DIM = 2;   // (x, y) coordinates
#ifdef LAYER2_ENA
const int HIDDEN1_DIM = 4;
const int HIDDEN2_DIM = 4;
#else
const int HIDDEN_DIM = 4;  // Tiny layer so we can see every single number
#endif
const int OUTPUT_DIM = 2;  // 2 classes: Red (0) or Blue (1)

struct DataPoint {
    float x;
    float y;
    int label; // INPUT_DIM: 0 for Red, 1 for Blue
};
