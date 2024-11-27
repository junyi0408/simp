#include "KdCurves4to3.h"

void KdCurves4to3::init()
{
	BO = Matrix<double, 10, 2>::Zero();
	BO.row(0) = BO.row(1) = C1.row(0);
	BO.row(8) = BO.row(9) = C4.row(3);
	BO1 = Matrix<double, 10, 2>::Zero();
	BO2 = Matrix<double, 10, 2>::Zero();
	BO1.row(1) = C1.row(1) - C1.row(0);
	BO2.row(8) = C4.row(2) - C4.row(3);
}

pair<Matrix<double, 10, 2>, double> KdCurves4to3::run()
{
	Matrix<double, 13, 2> C;
	C << C1, C2({ 1,2,3 }, all), C3({ 1,2,3 }, all), C4({ 1,2,3 }, all);
	double err = INFINITY;

	double a1 = BezierCurve(C1).ArcLenth();
	double a2 = BezierCurve(C2).ArcLenth();
	double a3 = BezierCurve(C3).ArcLenth();
	double a4 = BezierCurve(C4).ArcLenth();
	vector<double> tin = { a1 / (a1 + a2 + a3 + a4),(a1 + a2) / (a1 + a2 + a3 + a4),(a1 + a2 + a3) / (a1 + a2 + a3 + a4) };
	vector<pair<double, double>> tout_opt_temp = { {tin[0],tin[1]},{tin[0],tin[2]},{tin[1],tin[2]},{1.0 / 3.0,2.0 / 3.0} };
	set<pair<double, double>> unique_elements(tout_opt_temp.begin(), tout_opt_temp.end());
	vector<pair<double, double>> tout_opt(unique_elements.begin(), unique_elements.end());

	int num_var = 7;
	if (isRing)
		num_var = 8;

	double E = 0.0;
	vector<double> t0;

	for (const pair<double, double>& tout : tout_opt)
	{
		vector<double> R = { tout.second / tout.first - 1,(1 - tout.second) / (tout.second - tout.first) };
		if (isRing)
			R.push_back((C4.row(3) - C4.row(2)).norm() / (C1.row(1) - C1.row(0)).norm());
		auto [S, B] = build_odd_system(vector<cd>(R.begin(), R.end()));
		vector<double> temp_tout = { tout.first,tout.second };
		E = objective_t2(C.cast<cd>(), vector<cd>(tin.begin(), tin.end()), vector<cd>(temp_tout.begin(), temp_tout.end()), B, S).first.real();
		if (E < err)
		{
			err = E;
			t0 = temp_tout;
		}
	}

	vector<double> old(tin);
	old.insert(old.end(), t0.begin(), t0.end());
	old.push_back(t0[1] / t0[0] - 1);
	old.push_back((1 - t0[1]) / (t0[1] - t0[0]));
	if (isRing)
		old.push_back((C4.row(3) - C4.row(2)).norm() / (C1.row(1) - C1.row(0)).norm());
	E = err;
	vector<double> neww(old);

	auto f1 = [&](const vector<cd>& t, const VectorXcd& X, const VectorXcd& W) -> MatrixXcd {
		return objective_t3(C.cast<cd>(), t, X, W);
		};
	auto f2 = [&](const vector<cd>& t) -> cd {
		return objective_t4(C.cast<cd>(), t);
		};
	auto f3 = [&](const VectorXd& vec) -> double {
		std::vector<cd> stdVec(vec.data(), vec.data() + vec.size());
		return f2(stdVec).real();
		};

	for (int i = 0; i < max_iter; i++)
	{
		old = neww;
		auto [X, W] = Utils::global_quadrature_points(vector<double>(old.begin(), old.begin() + 3), vector<double>(old.begin() + 3, old.begin() + 5));
		MatrixXd J = Utils::Jacobian(f1, vector<cd>(old.begin(), old.end()), X.cast<cd>(), W.cast<cd>());
		MatrixXd dY = f1(vector<cd>(old.begin(), old.end()), X.cast<cd>(), W.cast<cd>()).real();
		VectorXd delta = J.completeOrthogonalDecomposition().pseudoInverse() * dY.reshaped();
		VectorXd df2dt = Eigen::VectorXd::Zero(num_var);
		for (int i = 0; i < num_var; i++)
		{
			vector<cd> temp(old.begin(), old.end());
			temp[i] += cd(0, 1e-100);
			df2dt(i) = f2(temp).imag() / 1e-100;
		}
		auto [alpha, newtemp] = Utils::backtracking_line_search(f3, Eigen::Map<Eigen::VectorXd>(old.data(), old.size()), df2dt, -delta, 0.3, 0.5);
		neww = std::vector<double>(newtemp.data(), newtemp.data() + newtemp.size());
		if (alpha == 0)
		{
			std::cerr << "Line search failed." << std::endl;
			break;
		}
		if (abs(E - f2(vector<cd>(neww.begin(), neww.end())).real()) < grad_tol)
			break;
		if (E < E_tol)
			break;
		E = f2(vector<cd>(neww.begin(), neww.end())).real();
	}
	pair<cd, Matrix<cd, 10, 2>> res2 = objective_t1(C.cast<cd>(), vector<cd>(neww.begin(), neww.begin() + 3), vector<cd>(neww.begin() + 3, neww.begin() + 5), vector<cd>(neww.begin() + 5, neww.end()));

	return { res2.second.real(),res2.first.real() };
}

pair<MatrixXcd, MatrixXcd> KdCurves4to3::build_odd_system(const vector<cd>& R)
{
	MatrixXcd B, S = MatrixXcd::Zero(20, 10);
	if (!isRing)
	{
		B = BO.cast<cd>();
		S.col(0) = BO1.cast<cd>().reshaped();
		S.col(1) = BO2.cast<cd>().reshaped();
		S(2, 2) = S(3, 3) = S(5, 4) = S(6, 5) = S(12, 6) = S(13, 7) = S(15, 8) = S(16, 9) = 1.0;
		S(4, 2) = S(14, 6) = -R[0];
		S(7, 4) = S(17, 8) = -R[1];
		S(4, 3) = S(14, 7) = 1.0 + R[0];
		S(7, 5) = S(17, 9) = 1.0 + R[1];
	}
	else
	{
		B = MatrixXcd::Zero(10, 2);
		B({ 1,8,9 }, all) << C1.cast<cd>().row(0), C4.cast<cd>().row(3) + R[2] * C1.cast<cd>().row(0), C4.cast<cd>().row(3);
		S(1, 0) = S(2, 1) = S(3, 2) = S(5, 3) = S(6, 4) = S(11, 5) = S(12, 6) = S(13, 7) = S(15, 8) = S(16, 9) = 1.0;
		S(7, 3) = S(17, 8) = -R[1];
		S(4, 1) = S(14, 6) = -R[0];
		S(4, 2) = S(14, 7) = 1.0 + R[0];
		S(7, 4) = S(17, 9) = 1.0 + R[1];
		S(8, 0) = S(18, 5) = -R[2];
	}
	return { S,B };
}

tuple<cd, Eigen::Matrix<cd, 10, 10>, Matrix<cd, 10, 2>> KdCurves4to3::objective(const MatrixXcd& C, const MatrixXcd& D, const vector<cd>& tin, const vector<cd>& tout)
{
	vector<cd> temp(tin);
	temp.insert(temp.end(), tout.begin(), tout.end());
	set<cd, Utils::ComplexCompare> unique_elements(temp.begin(), temp.end());
	vector<cd> t(unique_elements.begin(), unique_elements.end());
	t.insert(t.begin(), 0.0);
	t.push_back(1.0);
	MatrixXcd intervals(t.size() - 1, 2);
	for (int i = 0; i < t.size() - 1; i++)
		intervals.row(i) << t[i], t[i + 1];
	VectorXcd mid_pts = intervals.rowwise().mean();

	vector<cd> tin_(tin);
	tin_.insert(tin_.begin(), 0.0);
	tin_.push_back(1.0);
	vector<cd> tout_(tout);
	tout_.insert(tout_.begin(), 0.0);
	tout_.push_back(1.0);

	cd E = 0.0, c = 0.0;
	Matrix<cd, 10, 10> H = Matrix<cd, 10, 10>::Zero();
	Matrix<cd, 10, 2> F = Matrix<cd, 10, 2>::Zero();

	for (int i = 0; i < mid_pts.size(); i++)
	{
		int n1 = Utils::findFirstGreaterThan(tin_, mid_pts(i)) - 1, n2 = Utils::findFirstGreaterThan(tout_, mid_pts(i)) - 1;
		Eigen::MatrixXcd C1 = C.block<4, 2>(3 * n1, 0);
		Eigen::MatrixXcd C2 = D.block<4, 2>(3 * n2, 0);
		auto [Hi,Fi,ci,Ei] = Utils::cubic_cubic_integrated_distance(
			intervals(i, 0), intervals(i, 1),
			1.0 / (tin_[n1 + 1] - tin_[n1]), -tin_[n1] / (tin_[n1 + 1] - tin_[n1]),
			C1,
			1.0 / (tout_[n2 + 1] - tout_[n2]), -tout_[n2] / (tout_[n2 + 1] - tout_[n2]),
			C2
		);
		cd w1 = 1.0 / (tin_[n1 + 1] - tin_[n1]);
		cd w2 = 1.0 / (tout_[n2 + 1] - tout_[n2]);
		cd wi = (w1 + w2) / 2.0;

		H.block<4, 4>(3 * n2, 3 * n2) += wi * Hi;
		F.block<4, 2>(3 * n2, 0) += wi * Fi;
		c += wi * ci;
		E += wi * Ei;
	}
	return make_tuple(E, H, F);
}

pair<cd, Matrix<cd, 10, 2>> KdCurves4to3::objective_t2(const MatrixXcd& C, const vector<cd>& tin, const vector<cd>& tout, const MatrixXcd& B, const MatrixXcd& S)
{
	cd E = INFINITY;
	Matrix<cd, 10, 2> D = Matrix<cd, 10, 2>::Zero();

	int count1 = count_if(tin.begin(), tin.end(), [](cd val) {return (val.imag() == 0 && (val.real() >= 1 || val.real() <= 0)); });
	int count2 = count_if(tout.begin(), tout.end(), [](cd val) {return (val.imag() == 0 && (val.real() >= 1 || val.real() <= 0)); });
	if (count1 != 0 || count2 != 0)
		return { E,D };

	auto [_, H, F] = objective(C, D, tin, tout);
	MatrixXcd HH = MatrixXcd::Zero(2 * H.rows(), 2 * H.cols());
	HH.block(0, 0, H.rows(), H.cols()) = H;
	HH.block(H.rows(), H.cols(), H.rows(), H.cols()) = H;
	VectorXcd V = (S.transpose() * HH * S).completeOrthogonalDecomposition().pseudoInverse() * (-S.transpose() * F.reshaped() - S.transpose() * HH * B.reshaped());
	if (!isRing)
	{
		if (V(0).imag() == 0)
			V(0) = max(V(0).real(), 0.0);
		if (V(1).imag() == 0)
			V(1) = max(V(1).real(), 0.0);
	}
	D.col(0) = (S * V + B.reshaped()).head(10);
	D.col(1) = (S * V + B.reshaped()).tail(10);
	E = get<0>(objective(C, D, tin, tout));
	return { E,D };
}

pair<cd, Matrix<cd, 10, 2>> KdCurves4to3::objective_t1(const MatrixXcd& C, const vector<cd>& tin, const vector<cd>& tout, const vector<cd>& r)
{
	int count1 = count_if(tin.begin(), tin.end(), [](cd val) {return (val.imag() == 0 && (val.real() >= 1 || val.real() <= 0)); });
	int count2 = count_if(tout.begin(), tout.end(), [](cd val) {return (val.imag() == 0 && (val.real() >= 1 || val.real() <= 0)); });
	int count3 = count_if(r.begin(), r.end(), [](cd val) {return (val.imag() == 0 && val.real() < 0); });
	if ((count1 + count2 + count3) != 0)
		return { INFINITY,Eigen::Matrix<cd, 10, 2>::Zero() };
	auto [S1,B1] = build_odd_system(r);
	return objective_t2(C, tin, tout, B1, S1);
}

MatrixXcd KdCurves4to3::objective_t3(const Matrix<cd, 13, 2>& C, const vector<cd>& t, const VectorXcd& X, const VectorXcd& W)
{
	vector<cd> t1 = { t[0],t[1],t[2] };
	vector<cd> t2 = { t[3],t[4] };
	vector<cd> r(t.begin() + 5, t.end());
	MatrixXcd Y = MatrixXcd::Zero(X.size(), 2);
	int count1 = count_if(t1.begin(), t1.end(), [](cd val) {return (val.imag() == 0 && (val.real() >= 1 || val.real() <= 0)); });
	int count2 = count_if(t2.begin(), t2.end(), [](cd val) {return (val.imag() == 0 && (val.real() >= 1 || val.real() <= 0)); });
	int count3 = count_if(r.begin(), r.end(), [](cd val) {return (val.imag() == 0 && val.real() <= 0); });
	if ((count1 + count2 + count3) != 0)
	{
		std::cerr << "tin/tout should be in [0,1], r should be positive!" << std::endl;
		return Y;
	}
	
	auto [S, B] = build_odd_system(r);
	MatrixXcd D = MatrixXcd::Zero(10, 2);
	auto [_,H,F] = objective(C, D, t1, t2);
	MatrixXcd HH = MatrixXcd::Zero(2 * H.rows(), 2 * H.cols());
	HH.block(0, 0, H.rows(), H.cols()) = H;
	HH.block(H.rows(), H.cols(), H.rows(), H.cols()) = H;
	VectorXcd V = (S.transpose() * HH * S).completeOrthogonalDecomposition().pseudoInverse() * (-S.transpose() * F.reshaped() - S.transpose() * HH * B.reshaped());
	if (!isRing)
	{
		if (V(0).imag() == 0)
			V(0) = max(V(0).real(), 0.0);
		if (V(1).imag() == 0)
			V(1) = max(V(1).real(), 0.0);
	}
	D.col(0) = (S * V + B.reshaped()).head(10);
	D.col(1) = (S * V + B.reshaped()).tail(10);

	vector<cd> tA(t1);
	tA.push_back(1.0);
	tA.insert(tA.begin(), 0.0);
	vector<cd> tB(t2);
	tB.push_back(1.0);
	tB.insert(tB.begin(), 0.0);

	for (int i = 0; i < X.size(); i++)
	{
		cd u = X(i), w = W(i);
		int iA = Utils::findFirstGreaterThan(tA, u) - 1;
		cd g_A = 1.0 / (tA[iA + 1] - tA[iA]);
		cd h_A = -tA[iA] / (tA[iA + 1] - tA[iA]);
		Vector2cd yA = Utils::evaluate(C({ 3 * iA,3 * iA + 1,3 * iA + 2,3 * iA + 3 }, all), g_A * u + h_A);

		int iB = Utils::findFirstGreaterThan(tB, u) - 1;
		cd g_B = 1.0 / (tB[iB + 1] - tB[iB]);
		cd h_B = -tB[iB] / (tB[iB + 1] - tB[iB]);
		Vector2cd yB = Utils::evaluate(D({ 3 * iB,3 * iB + 1,3 * iB + 2,3 * iB + 3 }, all), g_B * u + h_B);

		Y.row(i) = sqrt(w) * (yA - yB);
	}
	return Y;
}

cd KdCurves4to3::objective_t4(const MatrixXcd& C, const vector<cd>& t)
{
	vector<cd> t1 = { t[0],t[1],t[2] };
	vector<cd> t2 = { t[3],t[4] };
	vector<cd> r(t.begin() + 5, t.end());
	cd E = 0.0;
	int count1 = count_if(t1.begin(), t1.end(), [](cd val) {return (val.imag() == 0 && (val.real() > 1 - 1e-15 || val.real() < 1e-15)); });
	int count2 = count_if(t2.begin(), t2.end(), [](cd val) {return (val.imag() == 0 && (val.real() > 1 - 1e-15 || val.real() < 1e-15)); });
	int count3 = count_if(r.begin(), r.end(), [](cd val) {return (val.imag() == 0 && val.real() < 1e-15); });
	bool b4 = (t1[1].real() > t1[0].real()) && (t1[2].real() > t1[1].real());
	bool b5 = t2[1].real() > t2[0].real();
	if ((count1 + count2 + count3) != 0 || !b4 || !b5)
		return INFINITY;
	auto [S1, B1] = build_odd_system(r);
	return objective_t2(C, t1, t2, B1, S1).first;
}