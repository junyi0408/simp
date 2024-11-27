#include "BezierCurve.h"

BezierCurve::BezierCurve(const vector<Vector2d>& p_CtrlPoints) {
	m_CtrlPoints = p_CtrlPoints;
	m_Degree = p_CtrlPoints.size() - 1;
}

BezierCurve::BezierCurve(const MatrixXd& p_CtrlPoints)
{
	int n = p_CtrlPoints.rows();
	vector<Vector2d> temp(n, Vector2d());
	for (int i = 0; i < n; i++)
		temp[i] = p_CtrlPoints(i, all);
	m_CtrlPoints = temp;
	m_Degree = n - 1;
}

MatrixXcd BezierCurve::GetCtrlPointsMatrix() const
{
	Eigen::MatrixXcd res;
	res.resize(m_CtrlPoints.size(), 2);
	for (int i = 0; i < m_CtrlPoints.size(); i++)
		res.row(i) = m_CtrlPoints[i];
	return res;
}

Vector2d BezierCurve::Evaluate(const double& para) const {
	double t = para;
	//assert(t >= 0.0 && t <= 1.0);
	//de Casteljau Ëã·¨
	vector<Vector2d> temp_points = m_CtrlPoints;

	for (int i = 1; i <= m_Degree; ++i) {
		for (int j = 0; j <= m_Degree - i; ++j) {

			temp_points[j] = temp_points[j] * (1.0 - t) + temp_points[j + 1] * t;

		}
	}
	return temp_points[0];
}

//void BezierCurve::deCasteljau(BezierCurve& left, BezierCurve& right, OpenMesh::Vec2d& vec, double t)
//{
//	OpenMesh::Vec2d p0 = m_CtrlPoints[0];
//	OpenMesh::Vec2d p1 = m_CtrlPoints[1];
//	OpenMesh::Vec2d p2 = m_CtrlPoints[2];
//	OpenMesh::Vec2d p3 = m_CtrlPoints[3];
//	std::vector<OpenMesh::Vec2d> left_ctrlPoint = { p0,(1 - t) * p0 + t * p1,(1 - t) * (1 - t) * p0 + 2 * t * (1 - t) * p1 + t * t * p2,Evaluate(t) };
//	std::vector<OpenMesh::Vec2d> right_ctrlPoint = { Evaluate(t),(1 - t) * (1 - t) * p1 + 2 * t * (1 - t) * p2 + t * t * p3,(1 - t) * p2 + t * p3,Evaluate(t) };
//	left = BezierCurve(left_ctrlPoint);
//	right = BezierCurve(right_ctrlPoint);
//}

double BezierCurve::ArcLenth() const
{
	std::function<double(double)> diff = [=](double t) 
		{
			Vector2d res = 3 * (1 - t) * (1 - t) * (m_CtrlPoints[1] - m_CtrlPoints[0]) +
				6 * (1 - t) * t * (m_CtrlPoints[2] - m_CtrlPoints[1]) +
				3 * t * t * (m_CtrlPoints[3] - m_CtrlPoints[2]);
			return res.norm();
		};
	return Utils::Gauss_Legendre_Quad(diff, 0, 1);
}
