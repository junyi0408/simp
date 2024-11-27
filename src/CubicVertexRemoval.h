#ifndef CUBICVERTEXREMOVAL_H
#define CUBICVERTEXREMOVAL_H
#include <Eigen/Dense>
#include <vector>
#include <set>
#include "CubicVertexRemovalC0.h"
using namespace std;
using namespace Eigen;
typedef complex<double> cd;

class CubicVertexRemoval
{
public:
	CubicVertexRemoval() {};
	CubicVertexRemoval(const Matrix<double, 4, 2>& c1, const Matrix<double, 4, 2>& c2)
		: C1(c1), C2(c2) {};
	~CubicVertexRemoval() {};

private:
	Matrix<double, 4, 2> C1;
	Matrix<double, 4, 2> C2;
	double E_tol = 1e-15;
	double grad_tol = 1e-8;
	int max_iter = 100;

public:
	pair<Matrix<double, 4, 2>, double> run();
};

#endif // !CUBICVERTEXREMOVAL_H
