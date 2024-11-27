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
    Eigen::MatrixXd controlPoints; // ���Ƶ����
    Eigen::MatrixXi indices;       // B��zier���ߵĿ��Ƶ���������

    QPointF bezierPoint(const Eigen::MatrixXd& P, const Eigen::VectorXi& indices, double t);
};

#endif // BEZIERWIDGET_H
