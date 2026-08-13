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

#include <QWidget>

#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPoint>
#include <QColor>
#include <QResizeEvent>
#include <QShowEvent>
#include <QTimer>
#include <QImage>
#include <vector>
#include <datatypes.h>
#include "classifier.h"


class PlotWidget : public QWidget {
 public:
    explicit PlotWidget(std::mt19937 &gen,
                        uint32_t seed,
                        QWidget* parent = nullptr);

 public slots:
    void onTimeout();

 protected:
    virtual void paintEvent(QPaintEvent* event) override;
    virtual void showEvent(QShowEvent *event) override;

 private:
    struct OutDataType {
        float A[OUTPUT_DIM];
    };

    QColor pix2color(float px);
    QColor filt2color(float k);
    QColor map2gray(float k);
    void drawInput(QPainter &painter, const DataPoint *indata);
    void drawFilters(QPainter &painter, int fnum, float *k, int kw, int kh);
    void drawActivationMap(QPainter &painter, int label, int fnum, float *A, int kw, int kh);
    void drawP(QPainter &p, std::list<OutDataType> &data, int x, int y, int width, int height);
    void drawProbabilities(QPainter &painter, int label, float *A, int sz);
    void saveToFile(QImage &img, int epoch);

 private:

    uint32_t seed_;
    std::mt19937 &gen_;
    std::list<OutDataType> P_[OUTPUT_DIM];
    QImage img_;
    CNNClassifier classifier_;
    QTimer *tmr_;
    float scalex_;
    float scaley_;
    int frameCnt_;      // for animation
};

