#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <unordered_set>
#define INF 1e10
#define EPSILON 1e-5


class Simplex{
public:
    /// 输入目标函数向量（按照单纯形表首行格式）以及约束增广矩阵
    Simplex(Eigen::RowVectorXd _tar, Eigen::MatrixXd _cstrn):
        target(_tar), constrain(_cstrn), _m(_cstrn.rows()), _n(_cstrn.cols() - 1), rhs(constrain.col(_n))
    {
        loop_cnt = 0;
        artifacts = 0;
        counter = new int[_m];
        memset(counter, 0, _m * sizeof(int));
    }

    Simplex(Eigen::MatrixXd _cstrn):
        constrain(_cstrn), _m(_cstrn.rows()), _n(_cstrn.cols() - 1), rhs(constrain.col(_n))
    {
        loop_cnt = 0;
        artifacts = 0;
        counter = new int[_m];
        memset(counter, 0, _m * sizeof(int));
    }

    ~Simplex(){
        delete[] counter;
    }
public:
    // 双阶段解法
    bool doubleStageSolve(const Eigen::RowVectorXd& tar);        
    bool solve();
    void showResults() const;
private:
    // 起始情况：首先从增广矩阵中找到m个线性无关向量（使用阶梯化的方式）
    // 此后取出这些列，拼合矩阵B，得到(B^-1)，继续使用对角化？
    // 化为典式，函数退出
    void getCanonical();   
    bool stageOneOptimize(std::unordered_set<int>& slct);        // 阶段一优化

    // 阶梯化，Binv为阶梯化结果，得到矩阵B
    void ladderize(Eigen::MatrixXd& B);
    bool findBiggestInspect(int& index) const;
    bool isLooping() const;
private:
    Eigen::RowVectorXd target;
    Eigen::MatrixXd constrain;
    int artifacts;                  // 人工变量个数
    int _m;
    int _n;                         // 此处_n的定义是 除了RHS列外的行数（输入的constrain包含RHS）
    int* counter;
    int loop_cnt;
    std::vector<int> base_index;
    const Eigen::Block<Eigen::MatrixXd, -1, 1, true>& rhs;
};

#endif  //__SIMPLEX_HPP__