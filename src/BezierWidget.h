#ifndef BEZIERWIDGET_H
#define BEZIERWIDGET_H

#include <QWidget>
#include <Eigen/Dense>

class BezierWidget : public QWidget {
    Q_OBJECT

public:
    explicit BezierWidget(const Eigen::MatrixXd& P, const Eigen::MatrixXi& I, QWidget* parent = nullptr);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    Eigen::MatrixXd controlPoints; // 控制点矩阵
    Eigen::MatrixXi indices;       // Bézier曲线的控制点索引矩阵

    QPointF bezierPoint(const Eigen::MatrixXd& P, const Eigen::VectorXi& indices, double t);
};

#endif // BEZIERWIDGET_H
