#include "BezierSimplifier.h"
#include "testFun.h"

static pair<MatrixXd, MatrixXi> reassemble(const vector<Component>& comps);

BezierSimplifier::BezierSimplifier(MatrixXd& P, MatrixXi& I, int target_num_curve, double G1_tol, double lossless_tol, double lossy_tol)
{
	this->P = P;
	this->I = I;
	this->target_num_curve = target_num_curve;
	this->G1_tol = G1_tol;
	this->lossless_tol = lossless_tol;
	this->lossy_tol = lossy_tol;
	init();
}

BezierSimplifier::~BezierSimplifier(){}

void BezierSimplifier::init()
{
	int n = I.rows();

	int j = 0;
	for (int i = 0; i < n-1; i++)
	{
		if (I(i,3) != I(i+1,0))
		{
			MatrixXi tempI = I.block(j, 0, i - j + 1, 4);
			j = i + 1;
			set<int> unique_elements(tempI.data(), tempI.data() + tempI.size());
			vector<int> unique_array(unique_elements.begin(), unique_elements.end());
			MatrixXd tempP = P(unique_array, all);
			components.push_back(Component(tempP, tempI.array() - tempI.minCoeff(), G1_tol));
		}
	}
	MatrixXi tempI = I.block(j, 0, n - j, 4);
	set<int> unique_elements(tempI.data(), tempI.data() + tempI.size());
	vector<int> unique_array(unique_elements.begin(), unique_elements.end());
	MatrixXd tempP = P(unique_array, all);
	components.push_back(Component(tempP, tempI.array() - tempI.minCoeff(), G1_tol));
}

void BezierSimplifier::run()
{
	int target_num_collapse = I.rows() - target_num_curve;

	Simplify2to1 sp(components, 1e-6, target_num_collapse);
	clock_t start = clock();
	sp.run();
	clock_t end = clock();
	double duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken: " << duration << " seconds" << std::endl;
	components = sp.GetComponents();


	auto [tempP, tempI] = getPI();
	clock_t start0 = clock();
	Simplify4to3 sp4to3(components, 10, tempI.rows() - target_num_curve);
	sp4to3.run();
	clock_t end0 = clock();
	double duration0 = double(end0 - start0) / CLOCKS_PER_SEC;
	std::cout << "Time taken: " << duration0 << " seconds" << std::endl;
	components = sp4to3.GetComponents();

	//auto [p, i] = reassemble(components);
	/*MatrixXd A = Utils::readSVG<MatrixXd>("../A.txt");
	MatrixXi B = Utils::readSVG<MatrixXi>("../B.txt").array() - 1;
	cout << p.rows() << endl;
	cout << "-------------------------------" << endl;
	cout << B << endl;
	cout << "haha" << endl;*/
	//cout << (i - B) << endl;
	//MatrixXi matrix = i - B;
	//vector<int> nonZeroRows;
	//for (int i = 0; i < matrix.rows(); ++i) {
	//	if (matrix.row(i).any()) {  // 如果这一行的任何元素非零
	//		nonZeroRows.push_back(i);
	//	}
	//}
}

pair<MatrixXd, MatrixXi> BezierSimplifier::getPI()
{
	return reassemble(components);
}

static pair<MatrixXd, MatrixXi> reassemble(const vector<Component>& comps)
{
	MatrixXd P;
	MatrixXi I;
	for (auto& comp : comps)
	{
		MatrixXd tempP = comp.P;
		MatrixXi tempI = comp.I;
		I.conservativeResize(I.rows() + tempI.rows(), 4);
		I.block(I.rows() - tempI.rows(), 0, tempI.rows(), tempI.cols()) = tempI.array() + P.rows();
		P.conservativeResize(P.rows() + tempP.rows(), 2);
		P.block(P.rows() - tempP.rows(), 0, tempP.rows(), tempP.cols()) = tempP;
	}
	return { P,I };
}