/// @brief 随机Kruscal算法生成迷宫
/// @由于我们使用4*4大小的画幅，其中涉及到奇偶地址操作
/// 地图大小为32 * 16的，需要改成31 * 15的会比较好（奇数特性）
#ifndef __KRUSCAL_H__
#define __KRUSCAL_H__
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
typedef unsigned char uint8_t;
typedef unsigned int uint;

// ===================== 栈模块(在搜索时需要使用) =======================//
typedef struct Stack{       // 大小为120的栈, uint8_t不够表示所有的地图位置
    uint8_t top;
    short data[200];
}stack;

/// @todo 初始化时188堵墙，计算是否正确？
extern uint8_t range;        // 取余范围
extern uint8_t backGround[512];   // 复用一下吧
extern uint rand_seed;              // 随机数种子
extern stack searches;              
extern short steps[4];
extern short walls[188];           // 需要检查或是打通的墙壁（此处需要检查）

void stackInit(stack* st);
void stackPush(stack* st, short val);
short stackPop(stack* st);
// ================= 栈模块结束 ==================//

void initialize();

/// @brief 随机选择墙
/// @param min_pos 输出，如果墙两侧地面不一致（代表两个连通支），则min_pos为标号更小的地面index值
/// @return 墙的index值
short randomSelWall(short* min_pos);

/// 破墙
void breakWalls();

/// 合并连通支（使用栈）
void mergeBranch(short wpos, uint8_t min_val);

#endif  //__KRUSCAL_H__