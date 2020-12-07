#-*-coding:utf-8-*-

import numpy as np
from copy import deepcopy as dcp
from collections import deque
import matplotlib.pyplot as plt

def greed(adjs, start):
    path = [start]
    now = start
    searched = {start}
    city_num = adjs.shape[0]
    for i in range(city_num - 1):
        min_val = float('inf')
        min_pos = -1
        for dst in range(city_num):
            if not dst in searched:
                val = adjs[now][dst]
                if val < min_val:
                    min_pos = dst
                    min_val = val
        searched.add(min_pos)
        path.append(min_pos)
        now = min_pos
    path.append(start)
    return path

def optimize(adjs, start):
    city = adjs.shape[0]
    result = greed(adjs, start)
    result.pop()
    result = exchange(adjs, result)
    temp = deque(result)
    temp.rotate(int(city / 2))
    result = list(temp)
    result = exchange(adjs, result)
    # 成环操作
    result.append(result[0])
    return result

def exchange(adjs, result):
    length=  len(result)
    for i in range(length - 3):
        for j in range(i + 2, length - 1):
            # 需要交换的情形
            x1 = result[i]
            x2 = result[(i + 1) % length]
            y1 = result[j % length]
            y2 = result[(j + 1) % length]
            if adjs[x1][x2] + adjs[y1][y2] > adjs[x1][y1] + adjs[x2][y2]:
                result[(i + 1) % length], result[j % length] = result[j % length], result[(i + 1) % length]
                result[(i + 2) % length:j % length] = result[(j - 1) % length : (i + 1) % length: -1]
    return result

def calcCost(adjs, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += adjs[path[i]][path[i + 1]]
    return cost

if __name__ == "__main__":
    start = 0
    city = 107
    adjs = np.fromfile(".//odom.bin")
    adjs = adjs.reshape((city, city))
    pos = np.fromfile(".//pos.bin")
    pos = pos.reshape((city, 2))
    min_cost = float('inf')
    min_path = []
    for i in range(city):
        order = optimize(adjs, start)
        c = calcCost(adjs, order)
        if c < min_cost:
            min_cost = c
            min_path = dcp(order)
            print("Smaller cost path with cost:", min_cost)
    min_path = ([12, 7, 42, 43, 44, 52, 100, 99, 102, 101, 103, 9, 3, 4, 2, 0, 1, 5, 8, 105, 104, 97, 98, 95, 96, 94, 89, 93, 74, 70, 90, 91, 92, 76, 69, 68, 66, 67, 72, 73, 71, 75, 27, 28, 19, 17, 18, 24, 25, 20, 21, 26, 22, 23, 10, 11, 30, 29, 106, 41, 40, 34, 33, 35, 37, 36, 38, 
    39, 32, 31, 15, 13, 14, 16, 65, 62, 61, 60, 53, 57, 58, 59, 82, 81, 80, 78, 79, 77, 85, 84, 88, 86, 83, 56, 87, 55, 54, 63, 64, 51, 46, 47, 45, 49, 48, 50, 6, 12])
    path = pos[min_path]
    print("The min path is:", min_path)
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 16
    plt.plot(path[:, 0], path[:, 1], c = 'k')
    plt.scatter(path[:, 0], path[:, 1], c = 'k', s = 7, label = '城市点位置')
    plt.title("近邻法 + 交换法结果示意图")
    plt.grid(axis = 'both')
    plt.legend()
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.show()