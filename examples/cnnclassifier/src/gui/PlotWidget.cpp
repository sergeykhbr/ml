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

#include "PlotWidget.h"
#include <QDir>

void generate_data(std::mt19937 &gen,
                   DataPoint *data);

PlotWidget::PlotWidget(std::mt19937 &gen,
                       uint32_t seed,
                       QWidget* parent) 
    : QWidget(parent),
    gen_(gen),
    classifier_(gen),
    img_(320, 240, QImage::Format_ARGB32_Premultiplied),
    seed_(seed)
{
    scalex_ = 0;
    scaley_ = 0;
    frameCnt_ = 0;
    img_.fill(Qt::white);
    tmr_ = new QTimer(this);
    connect(tmr_, &QTimer::timeout,
            this, &PlotWidget::onTimeout);
}

void PlotWidget::showEvent(QShowEvent *event) {
    tmr_->start(200);    // ms
}

QColor PlotWidget::pix2color(float px) {
    int rgb = 0;
    float p = std::max(0.0f, std::min(1.0f, px));
    rgb = static_cast<int>(255.0f * p);
    return QColor(qRgb(rgb, rgb, rgb)); // grey color
}

QColor PlotWidget::filt2color(float k) {
    int r = 0;
    int g = 0;
    int b = 0;
    if (k < 0) {
        b = static_cast<int>(-k * 169);
    } else {
        r = static_cast<int>(k * 169);
    }
    return QColor(qRgb(r, g, b));
}

void PlotWidget::drawInput(QImage &img, int epoch, const DataPoint *data) {
    QPainter painter(&img);

    painter.setRenderHint(QPainter::Antialiasing);

    QSize sz = size();
    int width = sz.width();
    int height = sz.height();

    int w0 = 0;
    int h0 = data->label * IMG_H;
    int scale = 4;
    QColor clr;
    for (int w = 0; w < IMG_W; w++) {
        for (int h = 0; h < IMG_H; h++) {
            clr = pix2color(data->pixels[w*IMG_W + h]);
            painter.setPen(clr);
            painter.setBrush(clr);
            //painter.drawPoint(w + w0, h + h0);
            painter.drawRect(scale*(w+w0), scale*(h+h0), scale, scale);
        }
    }

    /*// go through all pixels to find border:
    painter.setBrush(Qt::black);
    QPoint p;
    for (int w = 0; w < img.width(); w++) {
        for (int h = 0; h < img.height(); h++) {
            p = QPoint(w, h);
            painter.setPen(*frame);
            painter.drawPoint(p);
            frame++;
        }
    }

    // Draw generated dataset:
    painter.setPen(Qt::NoPen);           // No outline
    for (const auto& dot : dots_) {
        if (dot.label == 0) {
            painter.setBrush(Qt::red);
        } else if (dot.label == 1) {
            painter.setBrush(Qt::green);
        } else if (dot.label == 2) {
            painter.setBrush(Qt::blue);
        }
        //painter.drawEllipse(xy2point(dot.x, dot.y), 2, 2);
    }*/

    /*painter.setPen(Qt::darkGray);
    QString neurons = QString("Neurons: %1").arg(LAYER_DIM[0]);
    for (int i = 1; i < LAYER_NUM; i++) {
        neurons += QString(":%1").arg(LAYER_DIM[i]);
    }
    painter.drawText(80, 12*1, neurons);
    painter.drawText(80, 12*2, QString("Epoch: %1").arg(epoch));
    painter.drawText(80, 12*3, QString("Seed: %1").arg(seed_, 8, 16));
    painter.drawText(80, 12*4, QString("LearnRate: %1").arg(LEARNING_RATE, 0, 'f', 2));*/
}

void PlotWidget::drawFilters(QImage &img, int fnum, float *k, int kw, int kh) {
    QPainter painter(&img);
    
    int w0 = 60;
    int h0 = 0;
    int scale = 10;
    QColor clr;
    for (int f = 0; f < fnum; f++) {

        float *K = classifier_.getpFilter(f);
        h0 = f * (scale * kh + 5);
        for (int w = 0; w < kw; w++) {
            for (int h = 0; h < kh; h++) {
                clr = filt2color(K[w*kw + h]);
                painter.setPen(clr);
                painter.setBrush(clr);
                painter.drawRect(w0+scale*w, h0 + scale*h, scale, scale);
            }
        }
    }
}

void PlotWidget::drawProbabilities(QImage &img, int label, float *A, int sz) {
    QPainter painter(&img);
    QString str;
    for (int i = 0 ; i < sz; i++) {
        painter.setPen(Qt::gray);
        painter.setBrush(Qt::white);
        painter.drawRect(98, 10+20*i, 50, 20);  // erase text


        str = QString("P: %1").arg(A[i], 0, 'f', 2);
        if (i == label) {
            str += "*";
        }

        painter.setPen(Qt::black);
        painter.drawText(100, 20+20*i, str);
    }
}

void PlotWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    painter.drawImage(0, 0, img_);
    /*QImage *img = *std::next(listImages_.begin(), frameCnt_);
    painter.drawImage(0, 0, *img);

    if (++frameCnt_ >= listImages_.size()) {
        frameCnt_ = 0;
    }*/
}

void PlotWidget::onTimeout() {
    //std::cout << "--- Starting Training Optimization ---" << std::endl;
    
    DataPoint data;
    generate_data(gen_, &data);

    classifier_.trainStep(&data);
    drawInput(img_, 1, &data);
    drawFilters(img_, NUM_FILTERS, classifier_.getpFilter(0), KERNEL_SIZE, KERNEL_SIZE);
    drawProbabilities(img_, data.label, classifier_.getpResult(), OUTPUT_DIM);

    update();
}
