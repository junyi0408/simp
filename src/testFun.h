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
        // 遍历矩阵的每个元素并写入文件
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                outFile << matrix(i, j);
                if (j < matrix.cols() - 1) {
                    outFile << " ";  // 列之间用空格分隔
                }
            }
            outFile << endl;  // 行之间用换行分隔
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
        int sum = vec.sum();  // 使用Eigen的sum()函数计算向量的元素和
        outFile << sum << endl;  // 每个和写入文件后换行
    }

    outFile.close();
}


