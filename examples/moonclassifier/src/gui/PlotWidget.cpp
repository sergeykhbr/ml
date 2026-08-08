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

PlotWidget::PlotWidget(const std::vector<DataPoint>& dots,
                       MoonClassifier *classifier,
                       uint32_t seed,
                       QWidget* parent) 
    : QWidget(parent), dots_(dots), classifier_(classifier), seed_(seed)
{
    scalex_ = 0;
    scaley_ = 0;
    frameCnt_ = 0;
    tmr_ = new QTimer(this);
    connect(tmr_, &QTimer::timeout,
            this, &PlotWidget::onTimeout);
}

void PlotWidget::saveEpoch(int epoch) {
    QSize sz = size();
    int width = sz.width();
    int height = sz.height();
    float apperture = 2.0f*(1.0f + IN_DATA_DISTRIBUTION);      // -1.2 to +1.2
    scalex_ = static_cast<float>(width) / apperture;
    scaley_ = static_cast<float>(height) / apperture;

    float IN[INPUT_DIM];
    float OUT[OUTPUT_DIM];
    QPoint p;
    float *frame = new float [width * height];
    float *probability = frame;
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            p = QPoint(w, h);
            point2xy(p, IN);
            *probability = classifier_->forwardPass(IN, OUT);
            probability++;
        }
    }

    // save image:
    QDir dir(tr("screenshots"));
    if (!dir.exists()) {
        dir.mkpath(".");
    }
    QImage *img = new QImage(width, height, QImage::Format_ARGB32_Premultiplied);
    drawImage(*img, epoch, frame);
    QString filename = QString("frame_%1.png").arg(epoch, 4, 10, QChar('0'));
    QString fullname = dir.filePath(filename);
    img->save(fullname);

    listImages_.push_back(img);
    delete [] frame;

    // img2webp -loop 0 -d 200 frame_*.png -o sgd_l1_n4.webp
}

void PlotWidget::point2xy(QPoint &p, float *xy) {
    xy[0] = (p.x() / scalex_) - (1.0f + IN_DATA_DISTRIBUTION);
    xy[1] = (p.y() / scaley_) - (1.0f + IN_DATA_DISTRIBUTION);
}

QPoint PlotWidget::xy2point(float x, float y) {
    QPoint ret;
    ret.setX((x + (1.0f + IN_DATA_DISTRIBUTION))*scalex_);
    ret.setY((y + (1.0f + IN_DATA_DISTRIBUTION))*scaley_);
    return ret;
}

void PlotWidget::showEvent(QShowEvent *event) {
    tmr_->start(200);    // ms
}

QColor probability2color(float p) {
    p = std::max(0.0f, std::min(1.0f, p));
    int g = 0;
    int r = static_cast<int>(139.0f * p);
    int b = static_cast<int>(139.0f * (1.0f -  p));
    return QColor(qRgb(r, g, b));
}

void PlotWidget::drawImage(QImage &img, int epoch, float *frame) {
    img.fill(Qt::white);
    QPainter painter(&img);

    painter.setRenderHint(QPainter::Antialiasing);

    // go through all pixels to find border:
    painter.setBrush(Qt::black);
    QPoint p;
    float probability;
    for (int w = 0; w < img.width(); w++) {
        for (int h = 0; h < img.height(); h++) {
            p = QPoint(w, h);
            probability = *frame++;
            painter.setPen(probability2color(probability));
            painter.drawPoint(p);
        }
    }

    painter.setPen(Qt::darkGray);
#ifdef LAYER2_ENA
    painter.drawText(80, 12, QString("Neurons: %1:%2").arg(HIDDEN1_DIM).arg(HIDDEN2_DIM));
#else
    painter.drawText(80, 12, QString("Neurons: %1").arg(HIDDEN_DIM));
#endif
    painter.drawText(80, 24, QString("Epoch: %1").arg(epoch));
    painter.drawText(80, 36, QString("Seed: %1").arg(seed_, 8, 16));

    // Draw generated dataset:
    painter.setPen(Qt::NoPen);           // No outline
    for (const auto& dot : dots_) {
        if (dot.label) {
            painter.setBrush(Qt::red);
        } else {
            painter.setBrush(Qt::blue);
        }
        painter.drawEllipse(xy2point(dot.x, dot.y), 2, 2);
    }
}

void PlotWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    QImage *img = *std::next(listImages_.begin(), frameCnt_);
    painter.drawImage(0, 0, *img);

    if (++frameCnt_ >= listImages_.size()) {
        frameCnt_ = 0;
    }
}
