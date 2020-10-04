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
}

/**
 * 简单说明：检查就按照这个思路
 * 阶段一进入此函数前的工作：确定并记录人工变量（存于无序集合中）
 * 本函数需要解决的问题：消除所有人工变量，确定问题解的形式
 *      辅助问题是否有解？辅助问题无界等价于原问题无界，辅助问题最优解不是0则不需要再讨论原问题
 *      每次删除人工变量都是从无序集合中直接剔除，终止条件是：1.无界 2.检验数全为非正数 3.人工变量已经全部删除
 *      人工变量并不需要实际加入，只需要在出基时删除即可
 */
bool Simplex::stageOneOptimize(std::unordered_set<int>& slct){
    while (true){
        int ind = -1;
        if (findBiggestInspect(ind) == true){                                              // 确定入基变量
            std::cout << "Stage one optimal solution found with non-positive inspections.\n";
            break;
        }
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
        }
        std::cout << "In function [stageOneOptimize]: target and constrains are:\n";
        std::cout << "Target:\n";
        printMat(target);
        std::cout << "Constrains: \n";
        printMat(constrain);
        if (maxi_pos == -1){            // 对应Ai这一列所有元素全部小于0，也即x可以随意增大，问题无界
            std::cout << "No feasible result (from Stage One). Exiting...\n";
            return false;
        }
        else{                           // 确定了离基为 maxi_pos
            constrain.row(maxi_pos) /= column(maxi_pos);        // 本行转轴元归一
            for (int r = 0; r < _m; r++){
                if (column(r) != 0 && r != maxi_pos){           // 入基操作，对应列变为000...010..000
                    constrain.row(r) -= constrain.row(maxi_pos) * column(r);
                }
            }
            // 入基完成
            target -= constrain.row(maxi_pos) * target[ind];    // 将新的基变量检验数（即ind位置对应的检验数）通过行变换变为0
            const std::unordered_set<int>::iterator& iter = slct.find(maxi_pos);
            if (iter != slct.end()){        // 存在就删除
                slct.erase(iter);
            }
            else{
                std::cerr << "Error: Double deletion for artifacts.\n";
            }
            if (slct.size() < 1){
                std::cout << "Stage one ended with no more artifacts.\n";
                break;
            }
        }
    }
    const double& optimal = target(_n);
    if (optimal < EPSILON){
        std::cout << "Stage one optimal minimization: " << optimal << " (0.0). Feasible solution exists. Stage two is launched.\n";
        return true;
    }
    return false;
}

bool Simplex::doubleStageSolve(const Eigen::RowVectorXd& tar){
    std::unordered_set<int> selected;
    target = Eigen::RowVectorXd::Zero(_n + 1);              // 由于阶段一人工变量并不实际加入 对于原变量，全部设为0即可
    for (int col = 0; col < _n; col ++){
        const Eigen::Block<Eigen::MatrixXd, -1, 1, true>& column = constrain.col(col);
        int single = -1, break_flag = 0;
        for (int row = 0; row < _m; row ++){
            if (std::abs(column(row)) > EPSILON){
                if (single == -1){
                    single = row;
                }
                else{
                    break_flag = 1;
                    break;          // 说明一列中多个非0值
                }
            }
        }
        if (break_flag == 0 && single != -1){
            if (selected.find(single) == selected.end()){   // 未被加入
                if (column(single) > EPSILON){
                    constrain.row(single) /= column(single);
                    selected.emplace(single);
                }
                else if (column(single) < -EPSILON && std::abs(rhs(single)) < EPSILON){
                    constrain.row(single) *= -1.0;          // RHS位置为0并且单非零变量值小于0
                    selected.emplace(single);               // selected 是原问题本身就携带的单位列向量
                }
                std::cout << "Emplace" << single << " when iterating " << col << std::endl;
            }
        }
    }
    std::cout << "Original identity column number: " << selected.size() << std::endl;
    if (selected.size() == _m){                     // 原问题可以直接解，有_m个线性无关的单位列向量
        target = tar;
        return solve();
    }
    
    // 此数组用于判断哪些行对应了人工变量，对应人工变量的都需要出基
    for (int row = 0; row < _m; row ++){
        if (selected.find(row) == selected.end()){  // 没有加入的变量为人工变量
            target += constrain.row(row);           // 人工变量对应没有选中的行，则化人工变量（基）的检验数为0
        }
    }
    std::unordered_set<int> artis;
    for (int i = 0; i < _m; i++){
        if (selected.find(i) == selected.end()){
            artis.emplace(i);
        }
    }
    if (stageOneOptimize(artis) == true){
        target = tar;
        return solve();
    }
    else{
        std::cout << "No solution or bounds somehow.Double stage exiting...\n";
        return false;
    }
}

bool Simplex::solve(){
    getCanonical();
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
            if (column(r) > EPSILON && rhs(r) > -EPSILON){
                double delta = rhs(r) / column(r);
                if (delta < maxi){                  // 找到最大的改变量
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
            constrain.row(maxi_pos) /= column(maxi_pos);        // 本行转轴元归一
            for (int r = 0; r < _m; r++){
                if (column(r) != 0 && r != maxi_pos){           // 入基操作，对应列变为000...010..000
                    constrain.row(r) -= constrain.row(maxi_pos) * column(r);
                }
            }
            base_index[maxi_pos] = ind;       // 基索引存储更换
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
