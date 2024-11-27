#include "BezierWidget.h"
#include <QPainter>
#include <cmath>
#include <QPointF>

using namespace Eigen;

BezierWidget::BezierWidget(const Eigen::MatrixXd& P, const Eigen::MatrixXi& I, QWidget* parent)
    : QWidget(parent), controlPoints(P), indices(I) {
}

void BezierWidget::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // 获取窗口的大小
    double windowWidth = width();
    double windowHeight = height();

    // 获取控制点的边界
    double minX = controlPoints.col(0).minCoeff();
    double maxX = controlPoints.col(0).maxCoeff();
    double minY = controlPoints.col(1).minCoeff();
    double maxY = controlPoints.col(1).maxCoeff();

    // 计算控制点的宽度和高度
    double controlWidth = maxX - minX;
    double controlHeight = maxY - minY;

    // 计算缩放比例，使得控制点集适应窗口大小
    double scaleX = windowWidth / (controlWidth * 1.1);  // 留一些边距
    double scaleY = windowHeight / (controlHeight * 1.1);
    double scale = std::min(scaleX, scaleY);  // 保持比例一致

    // 获取控制点的中心位置
    double controlCenterX = (minX + maxX) / 2.0;
    double controlCenterY = (minY + maxY) / 2.0;

    // 获取窗口中心位置
    double centerX = windowWidth / 2.0;
    double centerY = windowHeight / 2.0;

    // 计算平移量，保证控制点在窗口中心
    double offsetX = centerX - controlCenterX * scale;
    double offsetY = centerY - controlCenterY * scale;

    // 绘制控制点
    painter.setPen(Qt::red);
    painter.setBrush(Qt::red);
    for (int i = 0; i < controlPoints.rows(); ++i) {
        double x = controlPoints(i, 0) * scale + offsetX;
        double y = controlPoints(i, 1) * scale + offsetY;
        painter.drawEllipse(QPointF(x, y), 3, 3); // 绘制控制点，使用小圆圈表示
    }

    // 绘制 Bézier 曲线
    painter.setPen(Qt::blue);
    for (int i = 0; i < indices.rows(); ++i) {
        Eigen::VectorXi rowIndices = indices.row(i);
        std::vector<QPointF> bezierPoints;

        for (double t = 0.0; t <= 1.0; t += 0.01) {
            QPointF point = bezierPoint(controlPoints, rowIndices, t);
            double x = point.x() * scale + offsetX;
            double y = point.y() * scale + offsetY;
            bezierPoints.push_back(QPointF(x, y));
        }

        // 使用 QPainter 绘制曲线
        for (size_t j = 0; j < bezierPoints.size() - 1; ++j) {
            painter.drawLine(bezierPoints[j], bezierPoints[j + 1]);
        }
    }
}


QPointF BezierWidget::bezierPoint(const Eigen::MatrixXd& P, const Eigen::VectorXi& indices, double t) {
    QPointF point(0, 0);
    int n = indices.size() - 1;

    for (int i = 0; i <= n; ++i) {
        double bernstein = std::pow(1 - t, n - i) * std::pow(t, i) * (std::tgamma(n + 1) / (std::tgamma(i + 1) * std::tgamma(n - i + 1)));
        point += bernstein * QPointF(P(indices[i], 0), P(indices[i], 1));
    }

    return point;
}
