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
        target(_tar), constrain(_cstrn), _m(_cstrn.rows()), _n(_cstrn.cols() - 1)
    {
        ;
    }
    ~Simplex();
public:
    bool solve();
private:
    // 起始情况：首先从增广矩阵中找到m个线性无关向量（使用阶梯化的方式）
    // 此后取出这些列，拼合矩阵B，得到(B^-1)，继续使用对角化？
    // 化为典式，函数退出
    void getCanonical();   

    // 阶梯化，Binv为阶梯化结果，得到矩阵B
    void ladderize(Eigen::MatrixXd& B);
    bool findBiggestInspect(int& index) const;
private:
    Eigen::RowVectorXd target;
    Eigen::MatrixXd constrain;
    int _m;
    int _n;
    std::vector<int> base_index;
};

void Simplex::getCanonical(){
    Eigen::MatrixXd B;
    Eigen::Matrix4d test;
    ladderize(B);
    Eigen::MatrixXd Binv = B.colPivHouseholderQr()
        .solve(Eigen::MatrixXd::Identity(B.rows(), B.cols()));
    constrain *= Binv;
    // 基变量检验数归0
    for (int index: base_index){
        if (target[index] != 0.0){
            target -= constrain.row(index) * target[index];     
        }
    }   // 此for循环结束后，对应得到典式
}

bool Simplex::solve(){
    getCanonical();
    while (true){
        const Eigen::Block<Eigen::MatrixXd, -1, 1, true>& rhs = constrain.col(_n);          // RHS
        int ind = -1;
        if (findBiggestInspect(ind) == false){                                              // 确定入基变量
            std::cout << "All inspections are non-positive. Optimal solution found.\n";
            return true;
        }
        // 开始调整基 首先需要在Aj中寻找正分量，确定最大改变量
        // 以下：ind表示的是入基的非基变量，maxi_pos表示的是出基的基变量代表的行数
        double maxi_pos = -1, maxi = INF;
        const Eigen::Block<Eigen::MatrixXd, -1, 1, true>& column = constrain.col(ind);
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
        else{                           // 确定了离基为 maxi_pos
            constrain.row(maxi_pos) /= column(maxi_pos);    // 本行转轴元归一
            for (int r = 0; r < _m; r++){
                if (column(r) != 0 && r != maxi_pos){       // 入基操作，对应列变为000...010..000
                    constrain.row(r) -= constrain.row(maxi_pos) * column(r);
                }
            }
            base_index[maxi_pos] = ind;                         // 基索引存储更换
            // 入基完成
            target -= constrain.row(maxi_pos) * target[ind];    // 将新的基变量检验数（即ind位置对应的检验数）通过行变换变为0
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
            base_index.emplace_back(col);
            start_row ++;
            if (start_row >= _m){
                return;
            }
        }
    }
}

bool Simplex::findBiggestInspect(int& index) const{
    bool all_minus = true;
    int maxi = EPSILON;
    for (int i = 0; i < _n; i++){
        const double& val = target[i];
        if (val > EPSILON){
            all_minus = false;
            if (val > maxi){
                index = i;
                maxi = val;
            }
        }
    }
    return all_minus;
}






#endif  //__SIMPLEX_HPP__