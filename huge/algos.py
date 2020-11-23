#-*-coding:utf-8-*-
#算法进阶实现

import numpy as np
import matplotlib.pyplot as plt
import copy as cp

#快速幂算法

def fastPower(base:int, expo:int):
    res = 1
    while expo:
        if (expo & 1) == 1:
            res *= base
        expo >>= 1
        base *= base
    return res

#传递闭包快速算法
class Warshall:
    #k每增加1，表示经中间点k的所有路径被加入
    #也就是说(i, 1), (1, j)若在路径中，下次可以使用(i, j)(如（i, 2）)
    #下次寻找的是(i, 2), (2, j),由上一行，(i, 2)可能由(i, 1), (1, 2)推来的
    #也就是说，(i, 2)，(2, j)推(i, j)实际是(i, 1), (1, 2), (2, j)推出的
    #这也就是二阶的间接关系
    #不同的是，warshall不是一次算出R^n,而是一次计算一个点为中间路径的情况
    #可能这一次,k=2时计算的情况有(i, 2), (i, j)的一阶间接关系，也有上述的二阶关系
    def __init__(self):
        pass

    def getBag(self, arr:list):
        size = len(arr)
        for k in range(size):                           
            for i in range(size):
                if arr[i][k]:                   #(i, k)存在， 若(k, j)也存在，则(i, j)存在
                    for j in range(size):
                        arr[i][j] = a[i][j] | arr[k][j]
        return arr
        
    @staticmethod        
    def printArr(arr:list):       #输出辅助
        size = len(arr)
        for i in range(size):
            print(arr[i])

#欧几里得辗转相除法
def gcd_Euclid(a:int, b:int):
    if a == b: return a
    A = max(a, b)
    B = A ^ a ^ b
    while(1):
        R = A % B
        if R == 0: return B
        A = B
        B = R

#Stein二进制位移gcd算法
def gcd_Stein(a:int, b:int):
    res = 1
    if a < b:               #保证a > b
        # = a^b
        #b = a^b
        #a = a^b
        a, b = b, a
    while (a & 1 == 0) and (b & 1 == 0):            #如果都是偶数就直接记录2这个因子
            a >>= 1
            b >>= 1
            res *= 2
    while(1):       
        while (a & 1 == 0):                 #如果一奇一偶，偶数直接除到没有2因子为止（因为奇数没有2因子）
            a >>= 1
        while (b & 1 == 0):
            b >>= 1
        a = abs(a - b)                          #相当于辗转相除，只不过这个是针对大数的，但是这个算法还是很不错
        if a == 0: return b * res
        b = min(a, b)
        

if __name__ == "__main__":
    ## warshall test
    #a = [[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    #b = cp.deepcopy(a)
    #w = Warshall()
    #a = w.getBag(a)
    #w.printArr(a)
    #print("\n")
    #w.printArr(b)
    #print(gcd_Euclid(423, 516))
    print(gcd_Stein(1023416, 1023416))
    print(gcd_Stein(98, 63))
    #exit()



