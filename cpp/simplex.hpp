#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#define INF 1e10
#define EPSILON 1e-5

class Simplex{
using Block = const Eigen::Block<Eigen::MatrixXd, 1, -1, false>&;
public:
    /// 输入目标函数向量（按照单纯形表首行格式）以及约束增广矩阵
    Simplex(Eigen::VectorXd _tar, Eigen::MatrixXd _cstrn):
        target(_tar), constrain(_cstrn), _m(_cstrn.rows()), _n(_cstrn.cols() - 1)
    {
        ;
    }
    ~Simplex();
public:
    // 起始情况：首先从增广矩阵中找到m个线性无关向量（使用阶梯化的方式）
    // 此后取出这些列，拼合矩阵B，得到(B^-1)，继续使用对角化？
    // 化为典式，函数退出
    void getInitial();   

    // 阶梯化，Binv为阶梯化结果，得到矩阵B
    void ladderize(Eigen::MatrixXd& B);

    bool solve();

    // 典式化 首先是乘以逆矩阵，此后将目标基变量变成0
private:
    Eigen::VectorXd target;
    Eigen::MatrixXd constrain;
    int _m;
    int _n;
    std::vector<int> baseIndex;
};

void Simplex::getInitial(){
    Eigen::MatrixXd B;
    Eigen::Matrix4d test;
    ladderize(B);
    Eigen::MatrixXd Binv = B.colPivHouseholderQr()
        .solve(Eigen::MatrixXd::Identity(B.rows(), B.cols()));
    constrain *= Binv;
    for (int index: baseIndex){
        if (target[index] != 0.0){
            target -= constrain.row(index) * target[index];     
        }
    }   // 此for循环结束后，对应得到典式
}

bool Simplex::solve(){
    bool all_minus = false;
    while (all_minus == false){
        all_minus = true;
        Block rhs = constrain.col(_n);          // RHS
        for (int i = 0; i < _n; i++){
            if (target[i] > EPSILON){                 // 此处有问题，需要找到最大的位置
                all_minus = false;
                // 开始调整基
                double maxi_pos = -1, maxi = INF;
                Block column = constrain.col(i);
                for (int r = 0; r < _m; r++){
                    if (column(r) > 0){
                        double delta = rhs(r) / column(r);
                        if (delta < maxi){
                            maxi = delta;
                            maxi_pos = r;
                        } 
                    }
                }
                if (maxi_pos == -1){            // 对应Ai这一列所有元素全部小于0，也即x可以随意增大，问题无界
                    std::cout << "No feasible result. Exiting...\n";
                    return false;
                }
                else{                           // 存在正分量
                    constrain.row(maxi_pos) /= column(maxi_pos);
                    for (int r = 0; r < _m; r++){
                        
                    }
                }
            }
        }
    }
}

void Simplex::ladderize(Eigen::MatrixXd& B){
    int start_row = 0;
    for (int col = 1; col < _n; col ++){
        int row = start_row;
        bool push_flag = false;
        if (constrain(col, col) == 0.0){
            for (; row < _m; row ++){
                if (constrain(row, col) != 0.0){
                    constrain.row(col).swap(constrain.row(row));        // 为0和不为0行进行交换
                    push_flag = true;
                    row ++;
                    break;
                }
            }
        }
        else{
            push_flag = true;
        }
        for (; row < _m; row ++){
            double head = constrain(row, col);
            if (constrain(row, col) != 0.0){
                constrain(row) -= head / constrain(col, col) * constrain(col);  // 消除头部0
            }
        }
        if (push_flag == true){
            baseIndex.emplace_back(col);
            start_row ++;
            if (start_row >= _m){
                return;
            }
        }
    }
}






#endif  //__SIMPLEX_HPP__