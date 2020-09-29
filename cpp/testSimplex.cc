#include <iostream>
#include "./include/simplex.hpp"

int main(){
    // 课本第14题
    Eigen::RowVectorXd cost(6);
    cost << 2, 1, 0, 0, 0, 0;
    Eigen::MatrixXd constrains(3, 6);
    constrains << 
        2, 5, 1, 0, 0, 60,
        1, 1, 0, 1, 0, 18,
        3, 1, 0, 0, 1, 44;
    std::cout << "Feeding data to simplex example...\n";
    Simplex sim(cost, constrains);
    std::cout << "Data feeding ended.\n";
    std::cout << "Starting to solve...\n";
    sim.solve();
    std::cout << "Solution completed.\n";
    sim.showResults();
    return 0;
}