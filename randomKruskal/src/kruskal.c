#include "../include/kruskal.h"

short steps[4] = {-1, 1, -32, 32};
uint8_t range = 188;
uint rand_seed = 0;              // 随机数种子
stack searches;
uint8_t backGround[512];   // 复用一下吧
uint8_t map[1024];
uint8_t xp = 1;
uint8_t yp = 1;
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

// 破墙，并生成可以使用的地图（直接绘制）
/// @todo map映射关系很容易错，查看一下
void breakWalls(){
    short wall, min_pos, i, j;
    uint8_t v1, v2;
    while(range > 0){                                       // 直到所有的墙全部取出
        wall = randomSelWall(&min_pos);
        if (wall != 0){
            backGround[wall] = backGround[min_pos];
            mergeBranch(wall, backGround[min_pos]);
        }
    }
    for (i = 0; i < 512; i++){          // 修复可能的问题
        if (backGround[i] > 1){
            backGround[i] = 1;
        }
    }
    for (j = 0; j < 16; j++){           // 由于不管是地面还是砖，都是 4 * 4大小的，所以j, j+1, j+2, j+3行都要设置
        for (i = 0; i < 16; i++){
            v1 = backGround[j * 32 + i * 2];
            v2 = backGround[j * 32 + i * 2 + 1];
            if (v1 & v2 == 1){          // 左右两块砖
                map[j * 64 + i] = 0xff;
                map[(4 * j + 1) * 16 + i] = 0xff;
                map[(4 * j + 2) * 16 + i] = 0xff;
                map[(4 * j + 3) * 16 + i] = 0xff;
            }
            else if (v1 == 1){          // 左边为地面
                map[j * 64 + i] = 0x0f;
                map[(4 * j + 1) * 16 + i] = 0x0f;
                map[(4 * j + 2) * 16 + i] = 0x0f;
                map[(4 * j + 3) * 16 + i] = 0x0f;
            }
            else if (v2 == 1){
                map[j * 64 + i] = 0xf0;
                map[(4 * j + 1) * 16 + i] = 0xf0;
                map[(4 * j + 2) * 16 + i] = 0xf0;
                map[(4 * j + 3) * 16 + i] = 0xf0;
            }
            else{                       // 全为地面
                map[j * 64 + i] = 0x00;
                map[(4 * j + 1) * 16 + i] = 0x00;
                map[(4 * j + 2) * 16 + i] = 0x00;
                map[(4 * j + 3) * 16 + i] = 0x00;
            }
        }
    }
    xp = 1;
    yp = 1;
    backGround[33] = 128;
}

void setPosOnGraph(uint8_t clear){
    uint8_t temp, val;
    if (xp & 1 == 1){           // x在奇数位置
        temp = (xp - 1) >> 1;   // 16列字节所在位置
        // 由于x在奇数位置，先将原有位置上的设置值清除
        if (clear){             // 清除位置
            map[temp + yp * 64] &= 0xf0;
            map[temp + (4 * yp + 1) * 16] &= 0xf0;
            map[temp + (4 * yp + 2) * 16] &= 0xf0;
            map[temp + (4 * yp + 3) * 16] &= 0xf0;
        }
        else{                   // 设置位置(2*2大小的方块代表玩家)
            map[temp + (4 * yp + 1) * 16] |= 0x06;
            map[temp + (4 * yp + 2) * 16] |= 0x06;
        }
    }
    else{                       // 偶数位置图像设置
        temp = xp >> 1;
        // 偶位置置零
        if (clear){             // 清除位置
            map[temp + yp * 64] &= 0x0f;
            map[temp + (4 * yp + 1) * 16] &= 0x0f;
            map[temp + (4 * yp + 2) * 16] &= 0x0f;
            map[temp + (4 * yp + 3) * 16] &= 0x0f;
        }
        else{                   // 设置位置
            map[temp + (4 * yp + 1) * 16] |= 0x60;
            map[temp + (4 * yp + 2) * 16] |= 0x60;
        }
    }
}

void printDebug(){
    short i, j;
    printf("This map is:\n");
    for (i = 0; i < 64; i++){
        for (j = 0; j < 16; j++){
            printf("%d, ", map[i * 16 + j]);
        }
        printf("\n");
    }
}

void move(uint8_t dir){
    uint8_t val;
    switch (dir)
    {
    case 0:
        if (xp < 29){       // 右移动
            if (backGround[xp + 1 + yp * 32] == 1){
                backGround[xp + yp * 32] = 1;
                setPosOnGraph(1);                       // 根据当前位置清除
                xp ++;
                backGround[xp + yp * 32] = 128;
                setPosOnGraph(0);                       // 根据当前位置设置
            }
        }
        break;
    case 1:
        if (xp > 1){       // 右移动
            if (backGround[xp - 1 + yp * 32] == 1){
                backGround[xp + yp * 32] = 1;
                setPosOnGraph(1);
                xp --;
                backGround[xp + yp * 32] = 128;
                setPosOnGraph(0);
            }
        }
        break;
    case 2:
        if (yp < 13){       // 右移动
            if (backGround[xp + (yp + 1) * 32] == 1){
                backGround[xp + yp * 32] = 1;
                setPosOnGraph(1);
                yp ++;
                backGround[xp + yp * 32] = 128;
                setPosOnGraph(0);
            }
        }
        break;
    default:
        if (yp > 1){       // 右移动
            if (backGround[xp + (yp - 1) * 32] == 1){
                backGround[xp + yp * 32] = 1;
                setPosOnGraph(1);
                yp --;
                backGround[xp + yp * 32] = 128;
                setPosOnGraph(0);
            }
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