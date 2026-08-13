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
static const int TITLE_SPACE_H = 24;

void generate_data(std::mt19937 &gen,
                   DataPoint *data);

PlotWidget::PlotWidget(std::mt19937 &gen,
                       uint32_t seed,
                       QWidget* parent) 
    : QWidget(parent),
    gen_(gen),
    classifier_(gen),
    img_(480, 240, QImage::Format_ARGB32_Premultiplied),
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
        b = static_cast<int>(-k * 255);
    } else {
        r = static_cast<int>(k * 255);
    }
    return QColor(qRgb(r, g, b));
}

QColor PlotWidget::map2gray(float k) {
    int rgb = 0;
    if (k < 0) {
        rgb = static_cast<int>(-k * 255);
    } else {
        rgb = static_cast<int>(k * 255);
    }
    return QColor(qRgb(rgb, rgb, rgb));
}

void PlotWidget::drawInput(QPainter &painter, const DataPoint *data) {
    int scale = 5;
    int w0 = 2;
    int h0 = TITLE_SPACE_H + data->label * (scale * IMG_H + 2);
    QColor clr;
    for (int w = 0; w < IMG_W; w++) {
        for (int h = 0; h < IMG_H; h++) {
            clr = pix2color(data->pixels[w*IMG_W + h]);
            painter.setPen(clr);
            painter.setBrush(clr);
            //painter.drawPoint(w + w0, h + h0);
            painter.drawRect(w0 + scale*w, h0 + scale*h, scale, scale);
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

void PlotWidget::drawFilters(QPainter &painter, int fnum, float *k, int kw, int kh) {
    int w0 = 80;
    int h0;
    int scale = 10;
    QColor clr;
    for (int f = 0; f < fnum; f++) {

        float *K = classifier_.getpFilter(f);
        h0 = TITLE_SPACE_H + 50 + f * (scale * kh + 5);
        for (int w = 0; w < kw; w++) {
            for (int h = 0; h < kh; h++) {
                clr = filt2color(K[w*kw + h]);
                painter.setPen(clr);
                painter.setBrush(clr);
                painter.drawRect(w0 + scale*w, h0 + scale*h, scale, scale);
            }
        }
    }
}

void PlotWidget::drawActivationMap(QPainter &painter, int label, int fnum, float *A, int kw, int kh) {
    int w0;
    int h0 = 0;
    int scale = 5;
    QColor clr;

    for (int f = 0; f < fnum; f++) {

        float *A = classifier_.getpActivationMap(f);
        h0 = TITLE_SPACE_H + 4 + label * (scale * kh + 12);
        w0 = 120 + f * (scale * kw + 2);
        for (int w = 0; w < kw; w++) {
            for (int h = 0; h < kh; h++) {
                clr = map2gray(A[w*kw + h]);
                painter.setPen(clr);
                painter.setBrush(clr);
                painter.drawRect(w0 + scale*w, h0 + scale*h, scale, scale);
            }
        }
    }
}

void PlotWidget::drawP(QPainter &p, std::list<OutDataType> &data, int x, int y, int width, int height) {
    int w = width;
    int h = height;
    QColor clr[OUTPUT_DIM] = {Qt::red, Qt::blue, Qt::green};

    p.setPen(Qt::gray);
    p.setBrush(Qt::white);
    p.drawRect(x, y, w, h);  // erase text

    p.setPen(Qt::black);
    // coordinate axis
    p.setPen(Qt::gray);
    p.setBrush(Qt::white);
    p.drawRect(x-20, y, 20, 12);  // erase text
    p.drawRect(x-20, y + h - 12, 20, 12);  // erase text
    p.setPen(Qt::black);
    p.drawText(x-18, y + 11, QString("1.0"));
    p.drawText(x-18, y + h-1, QString("0.0"));

    for (int i = 0; i < OUTPUT_DIM; i++) {
        int xmax = std::min((int)data.size(), w-1);
        int xstart = 0;
        int ystart = 0;
        int yend;
        p.setPen(clr[i]);
        for (auto &d : data) {
            yend = h - static_cast<int>(d.A[i] * (h - 1) + 0.5f);
            p.drawLine(x + xstart,
                       y + ystart,
                       x + xstart+1,
                       y + yend);
            xstart++;
            ystart = yend;
        }
    }
}

void PlotWidget::drawProbabilities(QPainter &painter, int label, float *A, int sz) {
    QString str;
    const int plot_w = 100;
    const int plot_h = 70;
    for (int i = 0 ; i < sz; i++) {

        //str = QString("P: %1").arg(P->A[i], 0, 'f', 2);
        if (i == label) {
            //str += "*";
            OutDataType P;
            for (int i = 0; i < OUTPUT_DIM; i++) {
                P.A[i] = A[i];
            }
            if (P_[i].size() < plot_w) {
                P_[i].push_back(P);
            }
            drawP(painter,
                  P_[i],
                  340,
                  TITLE_SPACE_H + i*(plot_h + 2),
                  plot_w,
                  plot_h);
        }
    }
}

void PlotWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    painter.drawImage(0, 0, img_);
}

void PlotWidget::saveToFile(QImage &img, int epoch) {
    // save image:
    QDir dir(tr("screenshots"));
    if (!dir.exists()) {
        dir.mkpath(".");
    }
    QString filename = QString("frame_%1.png").arg(epoch, 4, 10, QChar('0'));
    QString fullname = dir.filePath(filename);
    img.save(fullname);
}

void PlotWidget::onTimeout() {
    //std::cout << "--- Starting Training Optimization ---" << std::endl;
    
    DataPoint data;
    generate_data(gen_, &data);
    frameCnt_++;

    classifier_.trainStep(&data);


    QPainter painter(&img_);
    drawInput(painter, &data);
    drawFilters(painter, NUM_FILTERS, classifier_.getpFilter(0), KERNEL_SIZE, KERNEL_SIZE);
    drawActivationMap(painter, data.label, NUM_FILTERS, classifier_.getpActivationMap(0), OUT_W, OUT_H);
    drawProbabilities(painter, data.label, classifier_.getpResult(), OUTPUT_DIM);

    painter.setPen(Qt::white);
    painter.setBrush(Qt::white);
    painter.drawRect(0, 0, img_.width(), TITLE_SPACE_H);
    painter.setPen(Qt::black);
    QString title = QString("Epoch: %1").arg(frameCnt_);
    painter.drawText(2, 12, title);

    painter.drawText(72, 12, QString("Input:"));

    QString type[OUTPUT_DIM] = {"circle", "square", "triangle"};
    QColor clr[OUTPUT_DIM] = {Qt::red, Qt::blue, Qt::green};
    painter.setPen(clr[data.label]);
    painter.drawText(110, 12, type[data.label]);

    if (frameCnt_ < 210) {
        saveToFile(img_, frameCnt_);
    }
    update();
}
