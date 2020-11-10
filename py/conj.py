#-*-coding:utf-8-*-
"""
    PRP共轭梯度法 + Fibonacci搜索
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable as Var
from torch.autograd import grad

A = torch.FloatTensor([[1, 2, 4], [2, 3, -2], [4, -2, 6]]) 
b = torch.FloatTensor([3, 1, -5])
c = torch.FloatTensor([1, 2, 3])

sqr5 = torch.sqrt(5)

def getFibonacci(x):
    return 1 / sqr5 * ((0.5 + 0.5 * sqr5)**x - (0.5 - 0.5 * sqr5)**x)

class Conj:
    # func为pytorch自动求导定义下的函数
    # 线搜索最大迭代次数
    # self.func可能需要实现两套：输入为torch.tensor 与输入为 numpy.ndarray
    def __init__(self, func, max_ls_iter = 20):
        self.func = func
        self.max_ls_iter = max_ls_iter

    """
        func 为待估计函数
        d 为当前方向
        x 为初始点
        step_scl(scale) 步长
        max_iter 最大搜索次数
        返回一个步长值
        一般我们不会输入一个过小的d，否则在求单位向量的时候可能出现数值问题
    """
    @staticmethod
    def fibonacci(func, d, x, step_scl = 1, max_range = 16, max_iter = 16, prec = 1e-4):
        dn = np.linalg.norm(d)
        if dn < 1e-6:
            raise ValueError("Possible numeric problem detected. Norm too small.")
        direct = d / dn
        old_val = func(x)
        a = x.copy()
        b = torch.zeros_like(x)
        for i in range(1, max_range):
            now_val = func(x + i * direct * step_scl)
            if now_val >= old_val:
                b = x + i * direct * step_scl
                break
        else:
            raise RuntimeError("Interval initialization failed.")
        ratio = getFibonacci(max_iter - 2) / getFibonacci(max_iter)
        t1 = a + ratio * (b - a)
        t2 = b - ratio * (b - a)
        t1_val = func(t1)
        t2_val = func(t2)
        for i in range(max_iter):
            if t1_val <= t2_val:
                b = t2.copy()
                t2 = t1.copy()
                t1 = a + ratio * (b - a)
                t2_val = t1_val
                t1_val = func(t1)
            else:
                a = t1.copy()
                t1 = t2.copy()
                t2 = b - ratio * (b - a)
                t1_val = t2_val
                t2_val = func(t2)
            if np.linalg.norm(b - a) < prec:
                break
        return (a + b) / 2
    
    def solve(self, initial, criteria = 1e-3):
        x = Var(initial, requires_grad = True)
        y = self.func(x)
        ndim = x.size()[0]
        p = - grad(y, x, retain_graph = True)[0]       # 初始化
        last_g = - p.clone()
        i = 0
        while last_g.norm() >= criteria:
            x.data = Conj.fibonacci(self.func, p.numpy(), x.data.numpy())
            g = grad(y, x, retain_graph = True)[0]
            lamb = g.T @ (g - last_g) / (last_g.T @ last_g) # PRP（不依赖于完全的精确搜索（因为达不到））
            p = - g + lamb * p
            last_g = g.clone()
            i += 1
            if i >= ndim:   # 根据共轭梯度法，n次内收敛。假设不收敛需要重新开始一次
                p = - grad(y, x, retain_graph = True)[0]
                last_g = - p.clone()
                i = 0