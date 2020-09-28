#include <iostream>
#include <unordered_set>

int main(){
    std::unordered_set<int> contain;
    contain.emplace(3);
    contain.emplace(4);
    contain.emplace(5);
    contain.emplace(1);
    for (int i: contain){
        std::cout << i << ", ";
    }
    std::cout << std::endl;
    return 0;
}