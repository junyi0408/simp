#pragma once
#ifndef BEZIERSIMPLIFIER_H
#define BEZIERSIMPLIFIER_H
#include<Eigen/Dense>
#include<vector>
#include "Component.h"
#include "BezierCurve.h"
#include "Utils.h"
#include "Simplify2to1.h"
#include "Simplify4to3.h"
using namespace Eigen;
using namespace std;

class BezierSimplifier
{
public:
	MatrixXd P;
	MatrixXi I;
	int target_num_curve;
	int max_iter = 30;
	double G1_tol = 0.5;
	double lossless_tol = 1e-6;
	double lossy_tol = 10;
	std::string method = "gauss-newton";
	std::vector<Component> components;


public:
	BezierSimplifier() :P(MatrixXd()), I(MatrixXi()), target_num_curve(20), max_iter(30), G1_tol(0.0447),
		lossless_tol(1e-6), lossy_tol(10.0), method("gauss-newton"), components(std::vector<Component>(0)) {};
	BezierSimplifier(MatrixXd& P, MatrixXi& I, int target_num_curve, double G1_tol, double lossless_tol, double lossy_tol);
	~BezierSimplifier();

public:
	void init();
	void run();
	pair<MatrixXd, MatrixXi> getPI();
};
#endif // BEZIERSIMPLIFIER_H
