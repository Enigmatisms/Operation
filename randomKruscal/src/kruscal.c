#include "../include/kruscal.h"

short steps[4] = {-1, 1, -32, 32};
uint8_t range = 188;
uint rand_seed = 0;              // 随机数种子
stack searches;
uint8_t backGround[512];   // 复用一下吧
short walls[188];           // 需要检查或是打通的墙壁（此处需要检查）

// ===================== 栈模块(在搜索时需要使用) =======================//
void stackInit(stack* st){
    for (st->top = 0; st->top < 200; st->top++){
        st->data[st->top] = 0xffff;  
    }
    st->top = 0;
}

void stackPush(stack* st, short val){
    if (st->top < 200){
        st->data[st->top] = val;
        st->top ++;
    }
}

short stackPop(stack* st){
    if (st->top > 0){
        short val = st->data[st->top - 1];
        st->data[st->top-1] = 0xffff;
        st->top --;
        return val;
    }
    return 0xffff;
}

// 注意墙壁的生成需要从偶地址开始
// 生成标号地图以及隔墙
void initialize(){
    int i, j, k = 1;
    range = 0;
    rand_seed = (uint)time(NULL);
    srand(rand_seed);
    for (i = 0; i < 1024; i++){
        backGround[i] = 0;
    }
    for (i = 1; i < 30; i++){
        for (j = 1; j < 14; j++){
            if ((i & 1) && (j & 1)){                // 全为奇数时
                backGround[j * 32 + i] = k;
                k += 1;                         // 空余位置进行标记
            }
            else if ((i & 1) || (j & 1)){          // 墙壁位置
                walls[range] = i + j * 32;      // 保存墙壁位置
                range ++;
            }
        }
    }
    stackInit(&searches);
}

/// @brief 随机选择墙
/// @param min_pos 输出，如果墙两侧地面不一致（代表两个连通支），则min_pos为标号更小的地面index值
/// @return 墙的index值
short randomSelWall(short* min_pos){
    int index, wall = 0, temp;
    rand_seed = rand();
    srand(rand_seed);
    index = rand_seed % range;
    wall = walls[index];                                    // 被选中的墙index不会再被遍历到，只需要覆盖即可
    range --;           
    walls[index] = walls[range];                        
    if (backGround[wall - 1] != 0 &&
        backGround[wall + 1] != 0 &&
        backGround[wall - 1] != backGround[wall + 1]        // 墙左右两边地面标号不一致（不连通）
    ){
        if (backGround[wall - 1] < backGround[wall + 1]){
            *min_pos = wall - 1;                             // 较小的地面标号位置为左边地面
        }
        else{
            *min_pos = wall + 1;
        }
        return wall;
    }
    else if (backGround[wall - 32] != 0 &&
        backGround[wall + 32] != 0 &&
        backGround[wall - 32] != backGround[wall + 32]      // 墙上下两边地面标号不一致（不连同）
    ){
        if (backGround[wall - 32] < backGround[wall + 32]){
            *min_pos = wall - 32;                             // 较小的地面标号位置为左边地面
        }
        else{
            *min_pos = wall + 32;
        }
        return wall;
    }
    *min_pos = 0;
    return 0;                                               // 如果不需要破墙，返回0（地面标号一致，墙两侧连通）
}

// 破墙
void breakWalls(){
    short wall, min_pos;
    while(range > 0){                                       // 直到所有的墙全部取出
        wall = randomSelWall(&min_pos);
        if (wall != 0){
            backGround[wall] = backGround[min_pos];
            mergeBranch(wall, backGround[min_pos]);
        }
    }
}

// 合并连通支（使用栈）
// 深度有限搜索，从需要破墙的位置开始，每次查找周围4个位置，如果为大标号，记录到栈中并先将标号变为小标号
// 直到栈中元素全部被取出
void mergeBranch(short wpos, uint8_t min_val){
    short i, top = wpos;
    uint8_t val;
    stackPush(&searches, wpos);
    while(searches.top > 0){
        top = stackPop(&searches);
        for (i = 0; i < 4; i++){
            if (top + steps[i] < 0){
                continue;
            }
            val = backGround[top + steps[i]];
            if (val > 0 && val > min_val){
                backGround[top + steps[i]] = min_val;
                stackPush(&searches, top + steps[i]);
            }
        }
    }
}
// ================= 栈模块结束 ==================//