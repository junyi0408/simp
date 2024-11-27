#ifndef KDCURVES3TO2_H
#define KDCURVES3TO2_H

#include <Eigen/Dense>
#include <vector>
#include "BezierCurve.h"
using namespace std;
using namespace Eigen;
typedef complex<double> cd;

class KdCurves3to2
{
public:
	KdCurves3to2() {};
	KdCurves3to2(const Matrix<double, 4, 2>& c1, const Matrix<double, 4, 2>& c2, const Matrix<double, 4, 2>& c3, bool isring)
		:C1(c1), C2(c2), C3(c3), isRing(isring) {
		init();
	};
	~KdCurves3to2() {};

private:
	Matrix<double, 4, 2> C1;
	Matrix<double, 4, 2> C2;
	Matrix<double, 4, 2> C3;
	bool isRing = false;
	double E_tol = 1e-15;
	double grad_tol = 1e-5;
	int max_iter = 30;

	Matrix<double, 7, 2> BO;
	Matrix<double, 7, 2> BO1;
	Matrix<double, 7, 2> BO2;
	void init();

private:
	pair<MatrixXcd, MatrixXcd> build_odd_system(const vector<cd>& R);
	pair<cd, Matrix<cd, 7, 2>> objective_t(const MatrixXcd& C, const vector<cd>& tin, cd tout, const MatrixXcd& B, const MatrixXcd& S);
	
	MatrixXcd objective_t1(const Matrix<cd, 10, 2>& C, const vector<cd>& t, const VectorXcd& X, const VectorXcd& W);
	pair<cd, Matrix<cd, 7, 2>> objective_t2(const MatrixXcd& C, const vector<cd>& t);
	tuple<cd, Eigen::Matrix<cd, 7, 7>, Matrix<cd, 7, 2>> objective(const MatrixXcd& C, const MatrixXcd& D, const vector<cd>& tin, cd tout);

public:
	pair<Matrix<double, 7, 2>, double> run();
};

#endif // !KDCURVES3TO2_H
