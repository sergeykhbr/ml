#include "PlotWidget.h"

PlotWidget::PlotWidget(const std::vector<DataPoint>& dots,
                       MoonClassifier *classifier,
                       QWidget* parent) 
    : QWidget(parent), dots_(dots), classifier_(classifier)
{
    scalex_ = 0;
    scaley_ = 0;
}

void PlotWidget::point2xy(QPoint &p, float &x, float &y) {
    x = (p.x() / scalex_) - 1.2f;
    y = (p.y() / scaley_) - 1.2f;
}

void PlotWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    QSize sz = size();

    float apperture = 2.4;      // -1.2 to +1.2
    scalex_ = static_cast<float>(sz.width()) / apperture;
    scaley_ = static_cast<float>(sz.height()) / apperture;
        
    // Improve rendering quality (smooth circles)
    painter.setRenderHint(QPainter::Antialiasing);

    for (const auto& dot : dots_) {
        if (dot.label) {
            painter.setBrush(Qt::red);
        } else {
            painter.setBrush(Qt::blue);
        }
        painter.setPen(Qt::NoPen);           // No outline
        painter.drawEllipse(QPoint((dot.x + 1.2)*scalex_, (dot.y + 1.2)*scaley_), 2, 2);
    }

    // go through all pixels to find border:
    painter.setBrush(Qt::black);
    float x, y;
    float probability;
    QPoint p;
    for (int w = 0; w < sz.width(); w++) {
        for (int h = 0; h < sz.height(); h++) {
            p = QPoint(w, h);
            point2xy(p, x, y);
            probability = classifier_->forwardPass(x, y);
            if (probability > 0.45f && probability < 0.55f) {
                painter.drawEllipse(QPoint(w,h), 2, 2);
            }
        }
    }
}
