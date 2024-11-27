#ifndef KDCURVES4TO3_H
#define KDCURVES4TO3_H

#include <Eigen/Dense>
#include <vector>
#include "BezierCurve.h"
using namespace std;
using namespace Eigen;
typedef complex<double> cd;

class KdCurves4to3
{
public:
	KdCurves4to3() {};
	KdCurves4to3(const Matrix<double, 4, 2>& c1, const Matrix<double, 4, 2>& c2, const Matrix<double, 4, 2>& c3, const Matrix<double, 4, 2>& c4, bool isring)
		:C1(c1), C2(c2), C3(c3), C4(c4), isRing(isring) {
		init();
	};
	~KdCurves4to3() {};

private:
	Matrix<double, 4, 2> C1;
	Matrix<double, 4, 2> C2;
	Matrix<double, 4, 2> C3;
	Matrix<double, 4, 2> C4;
	bool isRing = false;
	double E_tol = 1e-15;
	double grad_tol = 1e-5;
	int max_iter = 30;

	Matrix<double, 10, 2> BO;
	Matrix<double, 10, 2> BO1;
	Matrix<double, 10, 2> BO2;
	void init();

private:
	pair<MatrixXcd, MatrixXcd> build_odd_system(const vector<cd>& R);
	pair<cd, Matrix<cd, 10, 2>> objective_t2(const MatrixXcd& C, const vector<cd>& tin, const vector<cd>& tout, const MatrixXcd& B, const MatrixXcd& S);
	MatrixXcd objective_t3(const Matrix<cd, 13, 2>& C, const vector<cd>& t, const VectorXcd& X, const VectorXcd& W);
	cd objective_t4(const MatrixXcd& C, const vector<cd>& t);
	pair<cd, Matrix<cd, 10, 2>> objective_t1(const MatrixXcd& C, const vector<cd>& tin, const vector<cd>& tout, const vector<cd>& r);
	tuple<cd, Eigen::Matrix<cd, 10, 10>, Matrix<cd, 10, 2>> objective(const MatrixXcd& C, const MatrixXcd& D, const vector<cd>& tin, const vector<cd>& tout);

public:
	pair<Matrix<double, 10, 2>, double> run();
};
#endif // !KDCURVES4TO3_H
