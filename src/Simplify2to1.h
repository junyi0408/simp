#ifndef SIMPLIFY2TO1_H
#define SIMPLIFY2TO1_H

#include<vector>
#include<Eigen/Dense>
#include<queue>
#include"Component.h"
#include"CubicVertexRemovalC0.h"
#include"CubicVertexRemoval.h"
#include <omp.h>
using namespace Eigen;
using namespace std;

class Simplify2to1
{
public:
	Simplify2to1() :target_num_collapse(0) {};
	Simplify2to1(const vector<Component>& u_components, double u_fit_tol, int u_num)
		: components(u_components), fit_tol(u_fit_tol), target_num_collapse(u_num) {
		init();
	};
	~Simplify2to1() {};

private:
	vector<Component> components;
	double fit_tol = 1e-6;
	int target_num_collapse;
	vector<VectorXi> exist_list;
	vector<MatrixXi> ids;
	VectorXi imap;

	vector<int> update_key;
	vector<Matrix<double, 4, 2>> cached_C;

private:
	void init();
	
	// utils function
	pair<Matrix<double, 4, 2>, double> compute_cost(int i, int k, bool isloop);
	int local2global(int i, int k);
	pair<int, int> global2local(int gid);
	void do_collapse(int i);
	vector<int> get_neighbors(int i, const VectorXi& list, bool isloop);
	pair<int, int> getLeftRightNeighbor(int i, const VectorXi& list, bool isloop);
	
public:
	void run();
	vector<Component> GetComponents() { return components; };
};

#endif SIMPLIFY2TO1_H //!SIMPLIFY2TO1_H