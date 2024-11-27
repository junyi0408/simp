#include <iostream>
#include <fstream>
#include <vector>
#include "Component.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

static void writeVectorOfMatricesToTxt(const vector<MatrixXi>& matrices, const string& filename) {
    ofstream outFile(filename);

    if (!outFile.is_open()) {
        cerr << "Failed to open the file!" << endl;
        return;
    }

    for (const auto& matrix : matrices) {
        // ���������ÿ��Ԫ�ز�д���ļ�
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                outFile << matrix(i, j);
                if (j < matrix.cols() - 1) {
                    outFile << " ";  // ��֮���ÿո�ָ�
                }
            }
            outFile << endl;  // ��֮���û��зָ�
        }
    }

    outFile.close();
}

static void writeSumsOfVectorsToTxt(const vector<VectorXi>& vectors, const string& filename) {
    ofstream outFile(filename);

    if (!outFile.is_open()) {
        cerr << "Failed to open the file!" << endl;
        return;
    }

    for (const auto& vec : vectors) {
        int sum = vec.sum();  // ʹ��Eigen��sum()��������������Ԫ�غ�
        outFile << sum << endl;  // ÿ����д���ļ�����
    }

    outFile.close();
}


