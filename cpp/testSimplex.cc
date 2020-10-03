#include <iostream>
#include "./include/simplex.hpp"

int main(){
    // =================== 课本第14题 单阶段法 ==================
    // Eigen::RowVectorXd cost(6);
    // cost << 2, 1, 0, 0, 0, 0;
    // Eigen::MatrixXd constrains(3, 6);
    // constrains << 
    //     2, 5, 1, 0, 0, 60,
    //     1, 1, 0, 1, 0, 18,
    //     3, 1, 0, 0, 1, 44;
    // std::cout << "Feeding data to simplex example...\n";
    // Simplex sim(cost, constrains);
    // std::cout << "Data feeding ended.\n";
    // std::cout << "Starting to solve...\n";
    // sim.solve();
    // std::cout << "Solution completed.\n";
    // sim.showResults();
    // =================== 两阶段法 ====================
    // 检验数需要包括最后的一列
    // 为什么最大值需要使用原来的，不用加负号
    // =================== 书第17题（1） ===================
    // Eigen::RowVectorXd cost(8);
    // cost << 3, 4, 2, 0, 0, 0, 0, 0;
    // Eigen::MatrixXd constrains(3, 8);
    // constrains << 
    //     1, 1, 1, 1, 1, 0, 0, 30,
    //     1, 6, 1, -2, 0, 1, 0, 0,
    //     0, 1, 0, 0, 0, 0, -1, 4;
    // =================== 书第17题（4） ===================
    Eigen::RowVectorXd cost(5);
    cost << 2, -4, 5, -6, 0;
    Eigen::MatrixXd constrains(2, 5);
    constrains << 
        1, 4, -2, 8, 2,
        -1, 2, 3, 5, 1;
    Simplex sim(constrains);
    std::cout << "Start to perform two stage solution.\n";
    sim.doubleStageSolve(cost);
    std::cout << "Two stage solution ended. Prepare to display results.\n";
    sim.showResults();
    return 0;
}