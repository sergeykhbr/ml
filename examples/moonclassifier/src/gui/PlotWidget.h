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
    void point2xy(QPoint &p, float &x, float &y);
    QPoint xy2point(float x, float y);
    void drawImage(QImage &img, float *frame);

 private:
    uint32_t seed_;
    const std::vector<DataPoint> &dots_;
    std::list<float *> listBorders_;
    MoonClassifier *classifier_;
    QTimer *tmr_;
    float scalex_;
    float scaley_;
    int frameCnt_;      // for animation
};

