#ifndef SIMPLIFY4TO3_H
#define SIMPLIFY4TO3_H

#include<vector>
#include<Eigen/Dense>
#include<queue>
#include"Component.h"
#include"CubicVertexRemovalC0.h"
#include"KdCurves3to2.h"
#include"KdCurves4to3.h"
#include<omp.h>
using namespace Eigen;
using namespace std;

class Simplify4to3
{
public:
	Simplify4to3() :target_num_collapse(0) {};
	Simplify4to3(const vector<Component>& u_components, double u_fit_tol, int u_num)
		: components(u_components), fit_tol(u_fit_tol), target_num_collapse(u_num) {
		init();
	};
	~Simplify4to3() {};

private:
	vector<Component> components;
	double fit_tol = 1e-6;
	int target_num_collapse;
	vector<VectorXi> exist_list;
	vector<MatrixXi> ids;
	VectorXi imap;

	vector<int> update_key;
	vector<Matrix<double, 10, 2>> cached_C;

private:
	void init();
	pair<Matrix<double, 10, 2>, double> compute_cost(int i, int k, bool isloop) const;
	int local2global(int i, int k);
	pair<int, int> global2local(int gid);
	void do_collapse(int i);
	vector<int> get_neighbors(int i, const VectorXi& list, bool isloop);
	void update_ids_after_collapse(int i, int k, const VectorXi& list, bool isloop);
	void update_ids_after_collapse_close_degenerate(int i, int k, const VectorXi& list);

public:
	void run();
	vector<Component> GetComponents() { return components; };
};

#endif // !SIMPLIFY4TO3_H
