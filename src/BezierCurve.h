#pragma once
#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include <vector>
#include <cmath>
#include <map>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Utils.h"
using namespace std;
using namespace Eigen;
class BezierCurve
{
protected:
	int m_Degree;
	vector<Vector2d> m_CtrlPoints;

public:
	BezierCurve() :m_Degree(0), m_CtrlPoints(vector<Vector2d>(1)) {};
	BezierCurve(const vector<Vector2d>& p_CtrlPoints);
	BezierCurve(const MatrixXd& p_CtrlPoints);
	

public:
	vector<Vector2d> GetCtrlPoints() const { return m_CtrlPoints; }
	MatrixXcd GetCtrlPointsMatrix() const;
	double ArcLenth() const;

	//void deCasteljau(BezierCurve& left, BezierCurve& right, OpenMesh::Vec2d& vec, double param);

public:
	Vector2d Evaluate(const double& para) const;
};

#endif // BEZIERCURVE_H



