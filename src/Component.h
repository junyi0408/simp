#ifndef COMPONENT_H
#define COMPONENT_H

#include <vector>
#include <Eigen/Dense>
#include "BezierCurve.h"
using namespace Eigen;

class Component
{
public:
	Component() :G1_tol(0.5) {};
	Component(const MatrixXd& u_P, const MatrixXi& u_I, double u_G1_tol)
		:P(u_P), I(u_I), G1_tol(u_G1_tol) {
		init();
	};
	~Component() {};

public:
	bool is_loop = false;
	double G1_tol;
	MatrixXd P;
	MatrixXi I;
	std::vector<int> non_G1;

public:
	void init();
};

#endif // !COMPONENT_H


