#ifndef __PRINTS_HPP__
#define __PRINTS_HPP__

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

template <typename T, int m, int n>
void printMat(const Eigen::Matrix<T, m, n>& mat){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            std::cout << mat(i, j) << ", "; 
        }
        std::cout << std::endl;
    }
}

void printMat(const Eigen::MatrixXd& mat){
    for (int i = 0; i < mat.rows(); i++){
        for (int j = 0; j < mat.cols(); j++){
            std::cout << mat(i, j) << ", "; 
        }
        std::cout << std::endl;
    }
}

void printMat(const Eigen::RowVectorXd& mat){
    for (int j = 0; j < mat.cols(); j++){
        std::cout << mat(j) << ", "; 
    }
    std::cout << std::endl;
}

template <typename Ty = int>
void printVector(const std::vector<Ty>& vec){
    for (size_t i = 0; i < vec.size(); i++){
        std::cout << vec[i] << ", ";
    }
    std::cout << std::endl;
}

#endif  //__PRINTS_HPP__