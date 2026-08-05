#pragma once

#include <QWidget>

#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPoint>
#include <QColor>
#include <vector>
#include <datatypes.h>
#include "classifier.h"


class PlotWidget : public QWidget {
 public:
    explicit PlotWidget(const std::vector<DataPoint>& dots,
                        MoonClassifier *classifier,
                        QWidget* parent = nullptr);

 protected:
    void paintEvent(QPaintEvent* event) override;

 private:
    void point2xy(QPoint &p, float &x, float &y);

 private:
    const std::vector<DataPoint> &dots_;
    MoonClassifier *classifier_;
    float scalex_;
    float scaley_;
};

