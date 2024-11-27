#include "Utils.h"
#include <unsupported/Eigen/Polynomials>

double Utils::Gauss_Legendre_Quad(const function<double(double)>& f, double a, double b)
{
	vector<double> weights = { 0.236926885056189, 0.236926885056189, 0.478628670499366, 0.478628670499366, 0.568888888888889 };
	vector<double> nodes = { -0.906179845938664, 0.906179845938664, -0.538469310105683, 0.538469310105683, 0.0 };

    // 计算高斯积分的变换系数
    double midpoint = (b + a) / 2.0;
    double half_length = (b - a) / 2.0;

    // 积分值初始化
    double integral = 0.0;

    // Gauss-Legendre积分求和
    for (int i = 0; i < 5; ++i) {
        double x = midpoint + half_length * nodes[i]; // 将节点变换到[a, b]
        integral += weights[i] * f(x);
    }

    integral *= half_length; // 加上区间缩放因子
    return integral;
}

void Utils::removeUnreferenced(
    const MatrixXd& V,
    const MatrixXi& F,
    MatrixXd& RV,
    MatrixXi& IMF)
{
    if (F.size() == 0)
    {
        RV.resize(0, V.cols());
        IMF.resize(0, F.cols());
        return;
    }

    // Flatten F and sort to get unique vertex indices
    vector<int> flatF(F.data(), F.data() + F.size());
    sort(flatF.begin(), flatF.end());
    auto last = unique(flatF.begin(), flatF.end());
    flatF.erase(last, flatF.end());

    // Create a map from old indices to new indices
    unordered_map<int, int> oldToNew;
    for (int i = 0; i < flatF.size(); ++i)
    {
        oldToNew[flatF[i]] = i;
    }

    // Create RV with only the referenced vertices
    RV.resize(flatF.size(), V.cols());
    for (int i = 0; i < flatF.size(); ++i)
    {
        RV.row(i) = V.row(flatF[i]);
    }

    // Create IMF by mapping the old vertex indices in F to the new ones
    IMF.resize(F.rows(), F.cols());
    for (int i = 0; i < F.rows(); ++i)
    {
        for (int j = 0; j < F.cols(); ++j)
        {
            IMF(i, j) = oldToNew[F(i, j)];
        }
    }
}


tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, cd, cd> Utils::cubic_cubic_integrated_distance(
	cd x, cd y,
	cd g_C, cd h_C, const Eigen::MatrixXcd& P,
	cd g_D, cd h_D, const Eigen::MatrixXcd& Q)
{
	Eigen::Vector4cd w = { 0.347854845137453857373063949222,
		0.652145154862546142626936050778,
		0.652145154862546142626936050778,
		0.347854845137453857373063949222 };
	Eigen::Vector4cd a = { -0.861136311594052575223946488893,
		-0.339981043584856264802665759103,
		0.339981043584856264802665759103,
		0.861136311594052575223946488893 };

	Eigen::Vector4cd u = (a.array() + cd(1,0)) * (y - x) / 2.0;
	u = u.array() + x;
	w *= ((y - x) / 2.0);

	Eigen::Matrix<cd, 4, 2> C;
	auto evaluate = [P](cd t) ->Eigen::Vector2cd {
		return pow(cd(1, 0) - t, 3) * P.row(0) + cd(3, 0) * pow(cd(1, 0) - t, 2) * t * P.row(1) +
			cd(3, 0) * pow(t, 2) * (cd(1, 0) - t) * P.row(2) + pow(t, 3) * P.row(3);
		};
	C.row(0) = evaluate(u[0] * g_C + h_C);
	C.row(1) = evaluate(u[1] * g_C + h_C);
	C.row(2) = evaluate(u[2] * g_C + h_C);
	C.row(3) = evaluate(u[3] * g_C + h_C);

	Eigen::Vector4cd T = (g_D * u).array() + h_D;

	Eigen::MatrixXcd M,W,H,F;
	M.resize(4, 4);
	M << pow((1 - T.array()), 3), 3 * T.array() * pow((1 - T.array()), 2), 3 * pow(T.array(), 2) * (1 - T.array()), pow(T.array(), 3);
	W = w.asDiagonal();
	H = M.transpose() * W * M;
	F = -M.transpose() * W * C;
	
    cd c = 0.5 * (C.transpose() * W * C).trace();
	cd E = cd(-1,0);
	if (Q.rows() == 4)
	{
		E = 0.5 * (Q.transpose() * H * Q).trace() + (Q.transpose() * F).trace() + c;
	}
	
    return std::make_tuple(H, F, c, E);
}


Eigen::VectorXd Utils::cubic_vertex_removal_polyfun(const Eigen::Vector4d& in1, const Eigen::Vector4d& in2)
{
    double iC11 = in1(0);
    double iC12 = in1(1);
    double iC13 = in1(2);
    double iC14 = in1(3);
    double iC21 = in2(0);
    double iC22 = in2(1);
    double iC23 = in2(2);
    double iC24 = in2(3);

    double t2 = iC12 * iC22;
    double t3 = iC11 * iC11;
    double t4 = iC12 * iC12;
    double t5 = iC23 * iC23;
    double t6 = iC24 * iC24;
    double t7 = iC11 * iC13 * 2.0;
    double t8 = iC12 * iC13 * 2.0;
    double t9 = iC11 * iC14 * 6.0;
    double t10 = iC12 * iC13 * 6.0;
    double t11 = iC12 * iC14 * 6.0;
    double t14 = iC11 * iC21 * 6.0;
    double t15 = iC12 * iC21 * 6.0;
    double t16 = iC14 * iC23 * 6.0;
    double t17 = iC14 * iC24 * 6.0;
    double t18 = iC22 * iC23 * 2.0;
    double t19 = iC22 * iC24 * 2.0;
    double t20 = iC21 * iC23 * 6.0;
    double t21 = iC21 * iC24 * 6.0;
    double t22 = iC22 * iC23 * 6.0;
    double t23 = iC22 * iC24 * 6.0;
    double t24 = iC11 * iC13 * 8.0;
    double t29 = iC12 * iC14 * 24.0;
    double t30 = iC11 * iC14 * 30.0;
    double t31 = iC14 * iC23 * 18.0;
    double t32 = iC14 * iC24 * 18.0;
    double t33 = iC12 * iC21 * 24.0;
    double t34 = iC11 * iC21 * 30.0;
    double t35 = iC21 * iC23 * 18.0;
    double t36 = iC21 * iC24 * 18.0;
    double t37 = iC11 * iC14 * 60.0;
    double t38 = iC11 * iC21 * 60.0;
    double t39 = iC12 * iC24 * 60.0;
    double t12 = t2 * 2.0;
    double t13 = t2 * 3.0;
    double t25 = -t11;
    double t26 = -t2;
    double t28 = -t15;
    double t40 = -t24;
    double t27 = -t12;

    Eigen::VectorXd out1(12);

    out1.segment<4>(0) <<
        t3 * 24.0 + t6 * 24.0 - iC11 * iC24 * 48.0,
        t3 * -192.0 - t6 * 72.0 - t39 + iC11 * iC12 * 60.0 + iC11 * iC23 * 60.0 + iC11 * iC24 * 264.0 - iC23 * iC24 * 60.0,
        t3 * 676.0 + t4 * 36.0 + t5 * 36.0 + t6 * 76.0 - iC11 * iC12 * 420.0 - iC11 * iC23 * 324.0 - iC11 * iC24 * 608.0 + iC12 * iC23 * 72.0 + iC12 * iC24 * 276.0 + iC23 * iC24 * 180.0,
        t3 * -1371.0 - t4 * 216.0 - t5 * 108.0 - t6 * 33.0 - t9 - t14 + t17 + t21 + iC11 * iC12 * 1269.0 + iC11 * iC23 * 729.0 + iC11 * iC24 * 756.0 - iC12 * iC23 * 324.0 - iC12 * iC24 * 513.0 - iC23 * iC24 * 189.0;

    out1.segment<2>(4) <<
        t3 * 1758.0 + t4 * 545.0 + t5 * 113.0 + t6 * 6.0 + t7 - t16 + t19 - t20 + t25 + t28 + t30 - t32 + t34 - t36 - iC11 * iC12 * 2149.0 - iC11 * iC22 * 2.0 - iC11 * iC23 * 881.0 - iC11 * iC24 * 546.0 + iC12 * iC23 * 584.0 + iC12 * iC24 * 487.0 - iC13 * iC24 * 2.0 + iC23 * iC24 * 83.0,
        t3 * -1468.0 - t4 * 741.0 - t5 * 50.0 - t6 * 2.0 + t8 - t18 - t23 + t26 + t29 + t31 + t32 + t33 + t35 + t36 - t37 - t38 + t40 + iC11 * iC12 * 2209.0 + iC11 * iC22 * 9.0 + iC11 * iC23 * 615.0 + iC11 * iC24 * 231.0 - iC12 * iC23 * 532.0 - iC12 * iC24 * 244.0 + iC13 * iC23 + iC13 * iC24 * 5.0 - iC23 * iC24 * 18.0;

    out1.segment<2>(6) <<
        t3 * 786.0 + t4 * 573.0 + t5 * 12.0 - t10 + t13 - t17 - t21 + t22 + t23 - t31 - t35 + t37 + t38 + t39 - iC11 * iC12 * 1383.0 + iC11 * iC13 * 12.0 - iC12 * iC14 * 36.0 - iC11 * iC22 * 15.0 - iC12 * iC21 * 36.0 - iC11 * iC23 * 249.0 - iC11 * iC24 * 57.0 + iC12 * iC23 * 252.0 - iC13 * iC23 * 3.0 - iC13 * iC24 * 3.0 + iC23 * iC24 * 6.0,
        t3 * -246.0 - t4 * 237.0 - t5 * 4.0 + t10 + t16 - t19 + t20 - t22 + t27 + t29 - t30 + t33 - t34 + t40 + iC11 * iC12 * 485.0 + iC11 * iC22 * 10.0 + iC11 * iC23 * 56.0 + iC11 * iC24 * 9.0 - iC12 * iC23 * 56.0 - iC12 * iC24 * 7.0 + iC13 * iC23 * 2.0;

    out1.segment<4>(8) <<
        t3 * 26.0 + t4 * 35.0 + t7 - t8 + t9 + t14 + t18 + t25 + t27 + t28 - iC11 * iC12 * 59.0 - iC11 * iC23 * 6.0 - iC11 * iC24 + iC12 * iC23 * 4.0 + iC12 * iC24,
        t3 * 12.0 + t4 * 9.0 + t13 - iC11 * iC12 * 21.0 - iC11 * iC22 * 3.0,
        t3 * -6.0 - t4 * 5.0 + t26 + iC11 * iC12 * 11.0 + iC11 * iC22,
        t3 + t4 - iC11 * iC12 * 2.0;

    return out1;
}

double Utils::cubic_vertex_removal_g(const Eigen::Vector4d& in1, const Eigen::Vector4d& in2, double st)
{
    double iC11 = in1(0);
    double iC12 = in1(1);
    double iC13 = in1(2);
    double iC14 = in1(3);
    double iC21 = in2(0);
    double iC22 = in2(1);
    double iC23 = in2(2);
    double iC24 = in2(3);

    double t2 = -iC11;
    double t3 = -iC12;
    double t4 = -iC24;
    double t5 = st - 1.0;
    double t6 = 1.0 / st;
    double t7 = iC11 + t3;
    double t8 = iC23 + t4;
    double t9 = 1.0 / t5;
    double t10 = t6 * t7;
    double t12 = t8 * t9;
    double t11 = -t10;
    double t13 = st * t12;
    double t14 = -t12;
    double t15 = iC11 + t4 + t11 + t12;
    double t16 = st * t15;
    double t17 = -t16;
    double t18 = iC12 + t2 + t10 + t16;
    double t21 = iC24 + t2 + t10 + t13 + t14 + t16;
    double t19 = st * t18;
    double t22 = st * t21;
    double t20 = -t19;
    double t23 = t7 + t11 + t17 + t19 + t22;
    double t24 = st * t23;

    double sg = std::pow((iC13 + t3 + t19), 2) +
        std::pow((iC22 + t2 + t10 + t16 - t22), 2) +
        std::pow((t8 + t12 - t13), 2) +
        std::pow((iC12 - iC14 + t20 + t24), 2) +
        std::pow((iC12 - iC21 + t20 + t24), 2);

    return sg;
}

std::vector<double> Utils::find_real_roots_in_interval(const Eigen::VectorXd& coefficients)
{
    // Solve the polynomial using Eigen's PolynomialSolver
    Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;
    solver.compute(coefficients);

    // Get all roots (including complex ones)
    Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootsType roots = solver.roots();

    std::vector<double> real_roots_in_interval;
    // Iterate over the roots to filter out real roots within [0, 1]
    for (int i = 0; i < roots.size(); ++i) {
        if (roots[i].imag() == 0.0) {  // Check if the root is real
            double real_root = roots[i].real();
            if (real_root >= 0.0 && real_root <= 1.0) {  // Check if the root is in the interval [0, 1]
                real_roots_in_interval.push_back(real_root);
            }
        }
    }
    return real_roots_in_interval;
}

Vector2cd Utils::evaluate(const Matrix<cd, 4, 2>& C, cd para)
{
    Vector2cd P1 = C.row(0);
    Vector2cd P2 = C.row(1);
    Vector2cd P3 = C.row(2);
    Vector2cd P4 = C.row(3);
    return P1 * pow((1.0 - para), 3) + P2 * 3.0 * pow((1.0 - para), 2) * para + 3.0 * P3 * (1.0 - para) * pow(para, 2) + P4 * pow(para, 3);
}


pair<VectorXd, VectorXd> Utils::global_quadrature_points(const vector<double>& ta, const vector<double>& tb)
{
    vector<double> temp(ta);
    temp.insert(temp.end(), tb.begin(), tb.end());
    temp.push_back(1.0);
    temp.insert(temp.begin(), 0.0);
    set<double> temp_set(temp.begin(), temp.end());
    vector<double> t(temp_set.begin(), temp_set.end());
    Vector4d w0 = { 0.347854845137453857373063949222,
        0.652145154862546142626936050778,
        0.652145154862546142626936050778,
        0.347854845137453857373063949222 };
    Vector4d u0 = { -0.861136311594052575223946488893,
        -0.339981043584856264802665759103,
        0.339981043584856264802665759103,
        0.861136311594052575223946488893 };

    VectorXd X = VectorXd::Zero((t.size() - 1) * 4);
    VectorXd W = VectorXd::Zero((t.size() - 1) * 4);
    for (int i = 0; i < t.size() - 1; i++)
    {
        X({ 4 * i,4 * i + 1,4 * i + 2,4 * i + 3 }) = (u0.array() + 1) * (t[i + 1] - t[i]) / 2.0 + t[i];
        W({ 4 * i,4 * i + 1,4 * i + 2,4 * i + 3 }) = w0 * (t[i + 1] - t[i]) / 2.0;
    }
    return { X,W };
}

MatrixXd Utils::Jacobian(function<Eigen::MatrixXcd(const vector<cd>&, const VectorXcd&, const VectorXcd&)> f, const vector<cd>& a, const VectorXcd& X, const VectorXcd& W)
{
    double eps = 1e-100;
    MatrixXd J = MatrixXd::Zero(2 * X.size(), a.size());
    for (int i = 0; i < a.size(); i++)
    {
        std::vector<cd> temp(a);
        temp[i] += cd(0, eps);
        MatrixXd grad = f(temp,X,W).imag() / eps;
        J.col(i) = grad.reshaped();
    }
    return J;
}

