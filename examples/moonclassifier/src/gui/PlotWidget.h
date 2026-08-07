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
    explicit PlotWidget(const std::vector<DataPoint>& dots,
                        MoonClassifier *classifier,
                        uint32_t seed,
                        QWidget* parent = nullptr);

    void saveEpoch(int epoch);

 public slots:
    void onTimeout() { update(); }

 protected:
    virtual void paintEvent(QPaintEvent* event) override;
    virtual void showEvent(QShowEvent *event) override;

 private:
    void point2xy(QPoint &p, float *xy);
    QPoint xy2point(float x, float y);
    void drawImage(QImage &img, int epoch, float *frame);

 private:
    uint32_t seed_;
    const std::vector<DataPoint> &dots_;
    std::list<float *> listBorders_;
    std::list<QImage *> listImages_;
    MoonClassifier *classifier_;
    QTimer *tmr_;
    float scalex_;
    float scaley_;
    int frameCnt_;      // for animation
};

