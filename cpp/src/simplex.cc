#include "../include/simplex.hpp"
#include "../include/prints.hpp"

void Simplex::getCanonical(){
    Eigen::MatrixXd B(_m, _m);
    ladderize(B);

    
    std::cout << "Ladderization completed.\n";
    Eigen::MatrixXd Binv = B.colPivHouseholderQr()
        .solve(Eigen::MatrixXd::Identity(_m, _m));
    constrain = Binv * constrain;
    // 基变量检验数归0
    std::cout << "Target before process:\n";
    printMat(target);
    for (size_t i = 0; i < _m; i++){
        target -= constrain.row(i) * target(base_index[i]);
    }
    // for (int index: base_index){
    //     if (target[index] != 0.0){
    //         target -= constrain.row(index) * target[index];     
    //     }
    // }   // 此for循环结束后，对应得到典式
    std::cout << "Constrains:\n";
    printMat(constrain);
    std::cout << "Target after process:\n";
    printMat(target);
}

bool Simplex::solve(){
    getCanonical();
    const Eigen::Block<Eigen::MatrixXd, -1, 1, true>& rhs = constrain.col(_n);          // RHS
    // 无法应对循环情况
    while (true){
        int ind = -1;
        if (findBiggestInspect(ind) == true){                                              // 确定入基变量
            std::cout << "All inspections are non-positive. Optimal solution found.\n";
            return true;
        }
        // 开始调整基 首先需要在Aj中寻找正分量，确定最大改变量
        // 以下：ind表示的是入基的非基变量，maxi_pos表示的是出基的基变量代表的行数
        // 对入基列寻找最大改变量
        double maxi_pos = -1, maxi = INF;
        const Eigen::Block<Eigen::MatrixXd, -1, 1, true>& column = constrain.col(ind);      // 入基列
        for (int r = 0; r < _m; r++){
            if (column(r) > EPSILON && rhs(r) > EPSILON){
                double delta = rhs(r) / column(r);
                if (delta < maxi){                  // 找到最大的改变量
                    maxi = delta;
                    maxi_pos = r;
                } 
            }
            else if (rhs(r) <= EPSILON){
                printf("RHS(%d) has odd result: %f\n", r, rhs(r));
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
            counter[ind] ++;                                    // 入基次数自增
            loop_cnt++;
            if (isLooping()){
                std::cout << "Simplex method is looping. Exiting...\n";
                break;
            }
        }
    }
    return true;
}

void Simplex::ladderize(Eigen::MatrixXd& B){
    int start_row = 0;
    for (int col = 0; col < _n + 1; col ++){
        int row = start_row;
        bool push_flag = false;
        if (constrain(row, col) == 0.0){
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
            row ++;
            push_flag = true;
        }
        for (; row < _m; row ++){
            double head = constrain(row, col);
            if (constrain(row, col) != 0.0){
                constrain.row(row) -= head / constrain(col, col) * constrain.row(col);  // 消除头部0
            }
        }
        if (push_flag == true){
            base_index.emplace_back(col);
            start_row ++;
            if (start_row >= _m){
                break;
            }
        }
    }
    if (base_index.size() < _m - 1){
        std::cerr << "Error: Contrain is not a full-rank matrix in terms of rows.\n";
    }
    for (size_t i = 0; i < _m; i++){
        B.col(i) = constrain.col(base_index[i]);
    }
}

void Simplex::showResults() const{
    printf("Minimal solution via Simplex method (One-stage): %lf\n\n", target(_n));
    printf("For each x, they are:\n");
    for (int i = 0; i < _m; i++){
        printf("x%d = %lf, ", base_index[i], constrain(i, _n));
    }
    std::cout << std::endl;
}

bool Simplex::findBiggestInspect(int& index) const{
    bool all_minus = true;
    int maxi = EPSILON;
    for (int i = 0; i < _n; i++){
        const double& val = target(i);
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

bool Simplex::isLooping() const{
    if (loop_cnt < _m + 1){         // 循环次数过少时不判定
        return false;
    }
    for (int i = 0; i < _m; i++){
        if (counter[i] > 4){
            return true;
        }
    }
    return false;
}