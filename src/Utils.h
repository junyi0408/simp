// Utils.h
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <functional>
#include <sstream>
#include <set>
#include <assert.h>
#include <unordered_map>

using namespace std;
using namespace Eigen;
typedef complex<double> cd;

class Utils {
public:
    template <typename T>
    static T readSVG(const string& path)
    {
        std::ifstream file(path);
        assert(file.is_open());

        std::string line;
        std::vector<std::vector<double>> data; // 用于暂存数据

        // 逐行读取文件
        while (std::getline(file, line)) {
            std::istringstream stream(line);
            std::vector<double> row;
            double value;

            // 读取当前行的每个元素
            while (stream >> value) {
                row.push_back(value);
            }

            if (!row.empty()) {
                data.push_back(row);
            }
        }
        file.close();

        // 动态确定行列数
        int rows = data.size();
        int cols = rows > 0 ? data[0].size() : 0;

        // 将数据填充到Eigen矩阵
        T matrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = data[i][j];
            }
        }

        return matrix;
    }

    static double Gauss_Legendre_Quad(const function<double(double)>& f, double a, double b);

    static tuple<double, VectorXd> backtracking_line_search(
        const function<double(const VectorXd&)>& f,
        const VectorXd& x0,
        const VectorXd& dfx0,
        const VectorXd& dx,
        double alpha,
        double beta,
        int max_iter = 30
    )
    {
        assert(alpha > 0 && alpha < 0.5);
        assert(beta > 0 && beta < 1);

        double t = 1.0;
        double fx0 = f(x0);

        for (int iter = 0; iter < max_iter; ++iter) {
            VectorXd x = x0 + t * dx;
            double fx = f(x);

            // 判断是否满足 Armijo 条件
            if (fx <= fx0 + alpha * t * (dfx0.array()*dx.array()).sum())
                return tuple<double, VectorXd>(t, x);
            t *= beta;
        }
        return tuple<double, VectorXd>(0.0, x0);
    }

    static tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, cd, cd> cubic_cubic_integrated_distance(
        cd x, cd y,
        cd g_C, cd h_C, const Eigen::MatrixXcd& P,
        cd g_D, cd h_D, const Eigen::MatrixXcd& Q);


    static void removeUnreferenced(
        const MatrixXd& V,
        const MatrixXi& F,
        MatrixXd& RV,
        MatrixXi& IMF);

    template <typename T>
    static int findIDfirst(const std::vector<T>& vec, T val) {
        int res = -1;
        for (int j = 0; j < vec.size(); ++j) {
            if (vec[j] > val) {
                res = j;
                break;
            }
        }
        return res;
    }

    template <typename T>
    static int findIDlast(const std::vector<T>& vec, T val) {
        int res = -1;
        for (int j = 0; j < vec.size(); ++j) {
            if (vec[j] >= val) {
                res = j - 1;
                break;
            }
        }
        return res;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
        removeDuplicateRows(const Eigen::MatrixBase<Derived>& mat) {
        using Scalar = typename Derived::Scalar;
        using RowType = Eigen::Matrix<Scalar, 1, Derived::ColsAtCompileTime>;

        // 自定义比较器，用于 std::set
        struct RowComparator {
            bool operator()(const RowType& a, const RowType& b) const {
                for (int i = 0; i < a.size(); ++i) {
                    if (a[i] < b[i]) return true;
                    if (a[i] > b[i]) return false;
                }
                return false; // 相等时，不认为 a < b
            }
        };

        // 使用 std::set 存储唯一的行
        std::set<RowType, RowComparator> uniqueRows;
        for (int i = 0; i < mat.rows(); ++i) {
            uniqueRows.insert(mat.row(i));
        }

        // 构建去重后的矩阵
        Eigen::Matrix<Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime> dedupMat(uniqueRows.size(), mat.cols());

        int idx = 0;
        for (const auto& row : uniqueRows) {
            dedupMat.row(idx++) = row;
        }

        return dedupMat;
    }


    static Eigen::VectorXd cubic_vertex_removal_polyfun(const Eigen::Vector4d& in1, const Eigen::Vector4d& in2);
    static double cubic_vertex_removal_g(const Eigen::Vector4d& in1, const Eigen::Vector4d& in2, double st);
    static std::vector<double> find_real_roots_in_interval(const Eigen::VectorXd& coefficients);

    // 4to3
    static pair<VectorXd, VectorXd> global_quadrature_points(const vector<double>& ta, const vector<double>& tb);
    static MatrixXd Jacobian(function<Eigen::MatrixXcd(const vector<cd>&, const VectorXcd&, const VectorXcd&)> f, const vector<cd>& a, const VectorXcd& X, const VectorXcd& W);
    
    struct ComplexCompare {
        bool operator()(const std::complex<double>& lhs, const std::complex<double>& rhs) const {
            if (lhs.real() < rhs.real()) {
                return true;
            }
            else if (lhs.real() == rhs.real()) {
                return lhs.imag() < rhs.imag();
            }
            return false;
        }
    };
    static int findFirstGreaterThan(const vector<cd>& vec, const cd& a) {
        ComplexCompare compare;
        for (int i = 0; i < vec.size(); ++i) {
            if (compare(a, vec[i])) {  // a < vec[i] means vec[i] is greater than a
                return i;
                break;
            }
        }
        return -1;  // Return empty optional if no element is greater
    }


    static Vector2cd evaluate(const Matrix<cd, 4, 2>& C, cd para);

private:
    Utils() = default; // 构造函数设为私有，防止实例化
};

#endif // UTILS_H
