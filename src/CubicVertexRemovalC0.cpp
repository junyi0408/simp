#include "CubicVertexRemovalC0.h"

tuple<Matrix<double, 4, 2>, double, double> CubicVertexRemovalC0::run()
{
	if ((C1.row(0) - C1.row(1)).norm() < 1e-5)
		C1.row(1) = C1.row(2);
	if ((C2.row(2) - C2.row(3)).norm() < 1e-5)
		C2.row(2) = C2.row(1);
	double temp1 = BezierCurve(C1).ArcLenth();
	double temp2 = BezierCurve(C2).ArcLenth();
	double t1 = temp1 / (temp1 + temp2);

	MatrixXd S, B = MatrixXd::Zero(4, 2);
	if (continuity == "G1G1")
	{
		B.row(0) = B.row(1) = C1.row(0);
		B.row(2) = B.row(3) = C2.row(3);
		Eigen::Matrix<double, 4, 2> B1 = Eigen::Matrix<double, 4, 2>::Zero();
		Eigen::Matrix<double, 4, 2> B2 = Eigen::Matrix<double, 4, 2>::Zero();
		B1.row(1) = C1.row(1) - C1.row(0);
		B2.row(2) = C2.row(2) - C2.row(3);
		S.resize(8, 2);
		S << B1.reshaped(), B2.reshaped();
	}
	else if (continuity == "G1C0")
	{
		B.row(0) = B.row(1) = C1.row(0);
		B.row(3) = C2.row(3);
		Eigen::Matrix<double, 4, 2> B1 = Eigen::Matrix<double, 4, 2>::Zero();
		B1.row(1) = C1.row(1) - C1.row(0);
		Eigen::Matrix<double, 4, 2> B2 = Eigen::Matrix<double, 4, 2>::Zero();
		Eigen::Matrix<double, 4, 2> B3 = Eigen::Matrix<double, 4, 2>::Zero();
		B2(2, 0) = B3(2, 1) = 1;
		S.resize(8, 3);
		S << B1.reshaped(), B2.reshaped(), B3.reshaped();
	}
	else if (continuity == "C0G1")
	{
		B.row(0) = C1.row(0);
		B.row(2) = B.row(3) = C2.row(3);
		Eigen::Matrix<double, 4, 2> B1 = Eigen::Matrix<double, 4, 2>::Zero();
		Eigen::Matrix<double, 4, 2> B2 = Eigen::Matrix<double, 4, 2>::Zero();
		B1(1, 0) = B2(1, 1) = 1;
		Eigen::Matrix<double, 4, 2> B3 = Eigen::Matrix<double, 4, 2>::Zero();
		B3.row(2) = C2.row(2) - C2.row(3);
		S.resize(8, 3);
		S << B1.reshaped(), B2.reshaped(), B3.reshaped();
	}
	else if (continuity == "C0C0")
	{
		B.row(0) = C1.row(0);
		B.row(3) = C2.row(3);
		Eigen::Matrix<double, 4, 2> B1 = Eigen::Matrix<double, 4, 2>::Zero();
		Eigen::Matrix<double, 4, 2> B2 = Eigen::Matrix<double, 4, 2>::Zero();
		B1(1, 0) = B2(1, 1) = 1;
		Eigen::Matrix<double, 4, 2> B3 = Eigen::Matrix<double, 4, 2>::Zero();
		Eigen::Matrix<double, 4, 2> B4 = Eigen::Matrix<double, 4, 2>::Zero();
		B3(2, 0) = B4(2, 1) = 1;
		S.resize(8, 4);
		S << B1.reshaped(), B2.reshaped(), B3.reshaped(), B4.reshaped();
	}
	else
		std::cerr << "continuity unknown error!!!";
	auto f = [=](cd t1) -> pair<cd, Matrix<cd, 4, 2>>
		{
			return objective_t1(C1.cast<cd>(), C2.cast<cd>(), t1, B.cast<cd>(), S.cast<cd>());
		};
	auto fb = [=](const VectorXd& t1) -> double
		{
			return objective_t1(C1.cast<cd>(), C2.cast<cd>(), t1(0), B.cast<cd>(), S.cast<cd>()).first.real();
		};
	auto [_, C] = f(t1);
	for (int i = 0; i < max_iter; i++)
	{
		double dfdt1 = f(cd(t1, 1e-100)).first.imag() / 1e-100;
		if (abs(dfdt1) < grad_tol)
			break;
		double dt1 = 0.5 * (dfdt1 > 0 ? -1 : 1);
		auto [alpha, t1_vec] = Utils::backtracking_line_search(fb, VectorXd::Constant(1, t1), VectorXd::Constant(1, dfdt1), VectorXd::Constant(1, dt1), 0.3, 0.5);
		t1 = t1_vec(0);
		if (alpha == 0)
		{
			// std::cerr << "line search failed" << std::endl;
			break;
		}
		auto [E, C0] = f(t1);
		if (E.real() < E_tol)
			break;
		C = C0;
	}
	auto [_1, _2, _5, E1] = Utils::cubic_cubic_integrated_distance(0.0, t1, 1.0 / t1, 0.0, C1, 1.0, 0.0, C);
	auto [_3, _4, _6, E2] = Utils::cubic_cubic_integrated_distance(t1, 1.0, 1.0 / (1.0 - t1), -t1 / (1.0 - t1), C2, 1.0, 0.0, C);
	return make_tuple(C.real(), E1.real() + E2.real(), t1);
}

pair<cd, Matrix<cd, 4, 2>> CubicVertexRemovalC0::objective_t1(
	const Matrix<cd, 4, 2>& C1,
	const Matrix<cd, 4, 2>& C2,
	cd t1,
	const MatrixXcd& B,
	const MatrixXcd& S)
{
	cd E = INFINITY;
	Matrix<cd, 4, 2> C = Matrix<cd, 4, 2>();
	if (t1.imag() == 0 && (t1.real() >= 1 || t1.real() <= 0))
		return { E,C };
	tuple<cd, MatrixXcd, MatrixXcd, cd> res = objective(C1, C2, C, t1);
	Eigen::MatrixXcd H, F, HH;
	Eigen::VectorXcd V;
	H = get<1>(res);
	F = get<2>(res);
	HH.resize(2 * H.rows(), 2 * H.cols());   HH.fill(cd(0, 0));
	HH.block(0, 0, H.rows(), H.cols()) = H;
	HH.block(H.rows(), H.cols(), H.rows(), H.cols()) = H;
	V = (S.transpose() * HH * S).completeOrthogonalDecomposition().pseudoInverse() * (-S.transpose() * F.reshaped() - S.transpose() * HH * B.reshaped());
	if (continuity == "G1G1")
		V = V.real().cwiseMax(0) + V.imag() * cd(0, 1);
	else if (continuity == "G1C0")
		V(0) = cd(max(V(0).real(), 0.0), V(0).imag());
	else if (continuity == "C0G1")
		V(V.rows() - 1) = cd(max(V(V.rows() - 1).real(), 0.0), V(V.rows() - 1).imag());
	C.col(0) = (S * V + B.reshaped()).head(4);
	C.col(1) = (S * V + B.reshaped()).tail(4);
	E = get<0>(objective(C1, C2, C, t1));
	return { E,C };
}

tuple<cd, MatrixXcd, MatrixXcd, cd> CubicVertexRemovalC0::objective(const Matrix<cd, 4, 2>& C1, const Matrix<cd, 4, 2>& C2, const Matrix<cd, 4, 2>& C, cd t1)
{
	cd E = INFINITY, c = 0.0;
	MatrixXcd H, F;
	if (t1.imag() == 0 && (t1.real() >= 1 || t1.real() <= 0))
		return make_tuple(E, H, F, c);

	cd w1 = (1.0 + 1.0 / t1) / 2.0;
	cd w2 = (1.0 + 1.0 / (1.0 - t1)) / 2.0;
	auto [H1, F1, c1, E1] = Utils::cubic_cubic_integrated_distance(0.0, t1, 1.0 / t1, 0.0, C1, 1.0, 0.0, C);
	auto [H2, F2, c2, E2] = Utils::cubic_cubic_integrated_distance(t1, 1.0, 1.0 / (1.0 - t1), -t1 / (1.0 - t1), C2, 1.0, 0.0, C);
	E = w1 * E1 + w2 * E2;
	H = w1 * H1 + w2 * H2;
	F = w1 * F1 + w2 * F2;
	c = w1 * c1 + w2 * c2;
	return make_tuple(E, H, F, c);
}