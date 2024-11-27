#ifndef CUBICVERTEXREMOVALC0_H
#define CUBICVERTEXREMOVALC0_H

#include <Eigen/Dense>
#include <vector>
#include "BezierCurve.h"
using namespace std;
using namespace Eigen;
typedef complex<double> cd;

class CubicVertexRemovalC0
{
public:
	CubicVertexRemovalC0() {};
	CubicVertexRemovalC0(const Matrix<double, 4, 2>& c1, const Matrix<double, 4, 2>& c2, const string& con)
		: C1(c1), C2(c2), continuity(con) {};
	~CubicVertexRemovalC0() {};

private:
	Matrix<double, 4, 2> C1;
	Matrix<double, 4, 2> C2;
	string continuity = "";
	double E_tol = 1e-15;
	double grad_tol = 1e-8;
	int max_iter = 100;

public:
	tuple<Matrix<double, 4, 2>, double, double> run();
	pair<cd, Matrix<cd, 4, 2>> objective_t1(
		const Matrix<cd, 4, 2>& C1,
		const Matrix<cd, 4, 2>& C2,
		cd t1,
		const MatrixXcd& B,
		const MatrixXcd& S);
	tuple<cd, MatrixXcd, MatrixXcd, cd> objective(const Matrix<cd, 4, 2>& C1, const Matrix<cd, 4, 2>& C2, const Matrix<cd, 4, 2>& C, cd t1);
};

#endif // !CUBICVERTEXREMOVALC0_H
