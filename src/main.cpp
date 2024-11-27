#include <iostream>
#include "BezierSimplifier.h"
#include "BezierWidget.h"
#include <QApplication>
#include "Utils.h"
#include <ctime>

int main(int argc, char* argv[])
{
	std::string path1 = "../numbersP.txt";
	std::string path2 = "../numbersI.txt";

	Eigen::MatrixXd P = Utils::readSVG<Eigen::MatrixXd>(path1);
	Eigen::MatrixXi I = Utils::readSVG<Eigen::MatrixXi>(path2);
	I.array() -= 1;

	BezierSimplifier bs(P, I, 300, 0.5, 1e-6, 10);
	bs.run();

	MatrixXd A = Utils::readSVG<MatrixXd>("../numbersP322.txt");
	MatrixXi B = Utils::readSVG<MatrixXi>("../numbersI322.txt").array() - 1;
	auto [P0, I0] = bs.getPI();
    QApplication a(argc, argv);

    BezierWidget w(A, B);
    w.resize(800, 800);
    w.show();

    return a.exec();
}