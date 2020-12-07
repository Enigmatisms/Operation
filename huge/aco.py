#!/usr/bin/env python3
#-*-coding:utf-8-*-
# 蚁群算法TSP

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
from collections import deque
import random as rd
import profile
import multiprocessing as mtp


fig_num = 0

class Ant:
    """
        蚂蚁类: 可以设置初始点，循环重置，路径计算
    """
    def __init__(self, city, Q = 10):
        self.now_pos = 0
        self.start = 0
        self.path = []
        self.plausible = {i for i in range(city)}
        self.last_cost = 0
        self.Q = Q              # 信息素分泌因子
        self.city = city

    def initialize(self, pos_num):
        self.now_pos = rd.choice(range(pos_num))
        self.plausible.remove(self.now_pos)
        self.path.append(self.now_pos)
        self.start = self.now_pos

    def updatePos(self, pos):
        self.now_pos = pos
        self.path.append(pos)
        self.plausible.remove(pos)

    def reset(self):
        self.plausible = {i for i in range(self.city)}
        self.plausible.remove(self.start)
        self.path = [self.now_pos]
        self.now_pos = self.start
        self.last_cost = 0

    def calculate(self, adjs:np.array):
        for i in range(self.city):
            self.last_cost += adjs[self.path[i]][self.path[(i + 1) % self.city]]
        return self.last_cost, self.path

    # 作用于外激素矩阵
    def secrete(self, ph_mat):
        length = len(self.path)
        for i in range(length):
            ph_mat[self.path[i]][self.path[(i + 1) % length]] = self.Q / self.last_cost

    def back2Start(self):
        self.now_pos = self.start
        self.path.append(self.start)


class BasicACO:
    fig_num = 0
    def __init__(self, city = 107, ants = 60, max_iter = 500, init = None):
        self.a = 1.0          # 信息启发因子（越大随机性越弱）
        self.b = 4          # 期望启发因子（越大越容易局部收敛（启发过大））
        self.p = 0.6        # 信息素挥发因子
        self.ant_num = ants
        self.city = city
        self.adjs = np.fromfile(".\\odom.bin")
        self.adjs = self.adjs.reshape((city, city))
        self.ants = [Ant(city) for i in range(ants)]
        self.phm = np.ones((city, city)) / 100       # 外激素矩阵
        self.max_iter = max_iter

        self.shortest = []
        self.cost = float("inf")
        self.costs = []
        for i in range(ants):
            self.ants[i].initialize(city)

        self.prefix = self.adjs ** ( - self.b)
        self.probs = np.zeros_like(self.prefix)
        if init != None:
            for i in range(len(init) - 1):
                self.phm[init[i]][init[i + 1]] += 1.5

    # 输入蚂蚁的index 输出list 一个对应于index蚂蚁的周围可选路径的概率
    # 输出可选路径
    def choiceForAnt(self, index):
        ant = self.ants[index]
        pos = ant.now_pos
        pl = list(self.ants[index].plausible)
        # 列表生成式的profile结果非常差
        prob = self.probs[pos][pl]
        return pl, list(prob)

    # 只在全局周游一次结束后，才开始计算外激素挥发以及分泌
    def updateSecretion(self):
        self.phm *= self.p      # 挥发因子
        for i in range(self.ant_num):
            self.ants[i].secrete(self.phm)

    # 所有蚂蚁进行一次周游
    def randomWander(self, fnum):
        for _it in range(self.max_iter):        # 最外层循环（周游次数）
            self.probs = (self.phm ** self.a) * self.prefix
            for k in range(self.city):          # 周游循环
                for i in range(self.ant_num):   # 对每个蚂蚁进行循环
                    if k == self.city - 1:
                        self.ants[i].back2Start()
                        cost, path = self.ants[i].calculate(self.adjs)
                        if cost < self.cost:
                            self.cost = cost
                            self.shortest = dcp(path)
                    else:
                        pos, choice_vec = self.choiceForAnt(i)
                        # print(pos, choice_vec)
                        next_pos = rd.choices(pos, choice_vec, k = 1)[0]
                        self.ants[i].updatePos(next_pos)
            self.costs.append(self.cost)
            self.updateSecretion()
            for i in range(self.ant_num):
                self.ants[i].reset()
            print("Iter %d / %d"%(_it, self.max_iter))
        self.shortest = self.exchange(self.shortest)
        print("Result(%d):"%(fnum), self.shortest)
        print("Random wader(%d ants) for %d times completed."%(self.ant_num, self.max_iter))
        self.draw(fnum)
    
    def optimize(self, arr:list):
        result = dcp(arr)
        result.pop()
        result = self.exchange(result)
        temp = deque(result)
        temp.rotate(int(self.city / 2))
        result = list(temp)
        result = self.exchange(result)
        # 成环操作
        result.append(result[0])
        return result

    def exchange(self, result):
        length=  len(result)
        for i in range(length - 3):
            for j in range(i + 2, length - 1):
                # 需要交换的情形
                x1 = result[i]
                x2 = result[(i + 1) % length]
                y1 = result[j % length]
                y2 = result[(j + 1) % length]
                if self.adjs[x1][x2] + self.adjs[y1][y2] > self.adjs[x1][y1] + self.adjs[x2][y2]:
                    result[(i + 1) % length], result[j % length] = result[j % length], result[(i + 1) % length]
                    result[(i + 2) % length:j % length] = result[(j - 1) % length : (i + 1) % length: -1]
        return result

    def draw(self, fnum):
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 16
        _x = np.arange(len(self.costs))
        plt.figure(fnum)
        plt.plot(_x, self.costs, color = "black")
        plt.grid()
        plt.xlabel("蚁群周游次数")
        plt.ylabel("最短路径长度")
        plt.title("蚁群算法迭代情况")
        plt.savefig(".\\fig%d.png"%(fnum))
        print("Final cost: %f"%(self.costs[-1]))
        # plt.show()

if __name__ == "__main__":
    # aco = BasicACO(107, max_iter = 200, ants = 500)
    # aco.randomWander()
    # profile.run("aco.randomWander()")
    # aco.draw()
    order = ([47, 46, 51, 64, 63, 54, 55, 87, 56, 83, 86, 88, 84, 85, 77, 78, 79, 80, 81, 82, 59, 58, 57, 53, 60, 61, 62, 65, 16, 14, 13, 15, 31, 32, 39, 38, 36, 37, 35, 33, 34, 40, 41, 106, 29, 30, 11, 10, 26, 22, 23, 21, 20, 25, 24, 18, 17, 19, 28, 27, 75, 71, 73, 72, 67, 66, 68, 69, 76, 92, 91, 90, 70, 74, 93, 94, 96, 95, 89, 98, 97, 104, 105, 8, 5, 1, 0, 2, 4, 3, 9, 103, 102, 101, 99, 100, 44, 52, 43, 42, 7, 12, 6, 50, 48, 49, 45, 47])

    aco_pool = [BasicACO(107, max_iter = 64, ants = 128, init = order) for i in range(4)]
    proc_pool = []
    for i in range(3):
        pr = mtp.Process(target = aco_pool[i].randomWander, args = (fig_num, ))
        fig_num += 1
        pr.start()
        proc_pool.append(pr)
    aco_pool[3].randomWander(3)