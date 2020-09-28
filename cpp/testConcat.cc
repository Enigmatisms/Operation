#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

std::vector<int> baseIndex;

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

void triangularize(Eigen::MatrixXd& B){
    int _m = B.rows();
    int _n = B.cols();
    int start_row = 0;                                  // 每次从start_row开始判定
    for (int col = 0; col < _n; col ++){
        int row = start_row;
        bool push_flag = false;
        if (B(row, col) == 0.0){
            for (; row < _m; row ++){
                if (B(row, col) != 0.0){
                    B.row(col).swap(B.row(row));        // 为0和不为0行进行交换
                    push_flag = true;
                    row ++;
                    break;
                }
            }
        }
        else{
            row ++;
            push_flag = true;
        }
        for (; row < _m; row ++){
            double head = B(row, col);
            if (head != 0.0){
                B.row(row) -= head / B(col, col) * B.row(col);  // 消除头部0
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

int main(){
    std::cout << "==================== test ladderize ==================\n";
    int row = 7, col = 12;
    Eigen::MatrixXd mat(row, col);
    mat << 0, 4, -12, 2, -2, 5, 3, 2, 1 -4, -2, 0,
        3, -1, -6, -2, 8, 0, 0, 3, -1, 1, 1, 0,
        -1, -1, 6, 2, 0, 0, 1, -1, 0, -3, -3, -3,
        2, 4, -5, -6, -7, -2, 2, 3, 4, 5, 6, 7,
        1, 0, 2, 1, 3, 0, 6, -6, -2, -1, 0, 3, 
        7, 2, 9, -1, 0, 0, 0, 0, 0, 3, 2, 1, 
        2, 1, 8, 3, 6, 1, 2, 0, 2, 6, 0, 1;
    std::cout << "Before ladderize:\n";
    printMat(mat); 
    triangularize(mat);
    std::cout << "After ladderize:\n";
    printMat(mat);
    std::cout << "Base index:\n";
    for (int ind: baseIndex){
        std::cout << ind << ", ";
    }
    std::cout << std::endl;
    Eigen::MatrixXd B(row, row);
    for (int i = 0; i < row; i++){
        B.col(i) = mat.col(baseIndex[i]);
    }
    std::cout << "B:\n";
    printMat(B);
    Eigen::MatrixXd Binv = B.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(row, row));
    std::cout << "Binv:\n";
    printMat(Binv);
    Eigen::MatrixXd res = Binv * mat;
    std::cout << "Result:\n";
    printMat(res);
    std::cout << "B mul Binv:\n";
    printMat(B * Binv);
    return 0;
}