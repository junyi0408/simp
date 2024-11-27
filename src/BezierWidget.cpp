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

    // ��ȡ���ڵĴ�С
    double windowWidth = width();
    double windowHeight = height();

    // ��ȡ���Ƶ�ı߽�
    double minX = controlPoints.col(0).minCoeff();
    double maxX = controlPoints.col(0).maxCoeff();
    double minY = controlPoints.col(1).minCoeff();
    double maxY = controlPoints.col(1).maxCoeff();

    // ������Ƶ�Ŀ�Ⱥ͸߶�
    double controlWidth = maxX - minX;
    double controlHeight = maxY - minY;

    // �������ű�����ʹ�ÿ��Ƶ㼯��Ӧ���ڴ�С
    double scaleX = windowWidth / (controlWidth * 1.1);  // ��һЩ�߾�
    double scaleY = windowHeight / (controlHeight * 1.1);
    double scale = std::min(scaleX, scaleY);  // ���ֱ���һ��

    // ��ȡ���Ƶ������λ��
    double controlCenterX = (minX + maxX) / 2.0;
    double controlCenterY = (minY + maxY) / 2.0;

    // ��ȡ��������λ��
    double centerX = windowWidth / 2.0;
    double centerY = windowHeight / 2.0;

    // ����ƽ��������֤���Ƶ��ڴ�������
    double offsetX = centerX - controlCenterX * scale;
    double offsetY = centerY - controlCenterY * scale;

    // ���ƿ��Ƶ�
    painter.setPen(Qt::red);
    painter.setBrush(Qt::red);
    for (int i = 0; i < controlPoints.rows(); ++i) {
        double x = controlPoints(i, 0) * scale + offsetX;
        double y = controlPoints(i, 1) * scale + offsetY;
        painter.drawEllipse(QPointF(x, y), 3, 3); // ���ƿ��Ƶ㣬ʹ��СԲȦ��ʾ
    }

    // ���� B��zier ����
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

        // ʹ�� QPainter ��������
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
