#include "CubicVertexRemoval.h"

pair<Matrix<double, 4, 2>, double> CubicVertexRemoval::run()
{
	set<double> s1, s2;
	for (int i = 0; i < C1.rows(); i++)
	{
		s1.insert(C1(i, 0));
		s1.insert(C2(i, 0));
		s2.insert(C1(i, 1));
		s2.insert(C2(i, 1));
	}
	int index = s1.size() >= s2.size() ? 0 : 1;
	VectorXd coeff = Utils::cubic_vertex_removal_polyfun(C1.col(index), C2.col(index));
	vector<double> roots = Utils::find_real_roots_in_interval(coeff);
	double t = -1, val = INFINITY;
	for (double root : roots)
	{
		double temp = Utils::cubic_vertex_removal_g(C1.col(index), C2.col(index), root);
		if (temp < val)
		{
			val = temp;
			t = root;
		}
	}

	Matrix<double, 4, 2> C;
	if (t != -1)
	{
		C.row(0) = C1.row(0);
		C.row(1) = C1.row(0) + (C1.row(1) - C1.row(0)) / t;
		C.row(2) = C2.row(3) + (C2.row(2) - C2.row(3)) / (1 - t);
		C.row(3) = C2.row(3);
	}
	else
	{
		tuple<Matrix<double, 4, 2>, double, double> res = CubicVertexRemovalC0(C1, C2, "G1G1").run();
		C = get<0>(res);
		t = get<2>(res);
	}
	auto [_1, _2, _5, E1] = Utils::cubic_cubic_integrated_distance(0.0, t, 1.0 / t, 0.0, C1, 1.0, 0.0, C);
	auto [_3, _4, _6, E2] = Utils::cubic_cubic_integrated_distance(t, 1.0, 1.0 / (1.0 - t), -t / (1.0 - t), C2, 1.0, 0.0, C);
	return { C,E1.real() + E2.real() };
}