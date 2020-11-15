#-*-coding:utf-8-*-
"""
    PRP共轭梯度法 + Fibonacci搜索
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable as Var
from torch.autograd import grad
from lineSearch import fibonacci, draw2D

# A = torch.FloatTensor([[1, 2, 4], [2, 3, -2], [4, -2, 6]]) 
# b = torch.FloatTensor([3, 1, -5])
# c = torch.FloatTensor([1, 2, 3])

A = torch.FloatTensor([[1, 2], [2, 5]])
b = torch.FloatTensor([1, -2])
c = torch.FloatTensor([1])

sqr5 = np.sqrt(5)

def func_to_eval(x):
    if type(x) == np.ndarray:
        return float(0.5 * x.T @ A.numpy() @ x + (b.numpy()).T @ x + c.numpy())
    else:
        return 0.5 * x.T @ A @ x + b.T @ x + c

class Conj:
    # func为pytorch自动求导定义下的函数
    # 线搜索最大迭代次数
    # self.func可能需要实现两套：输入为torch.tensor 与输入为 numpy.ndarray
    def __init__(self, func, dim = 2, max_ls_iter = 16):
        self.func = func
        self.max_ls_iter = max_ls_iter
        self.n = dim
        self.pos = []       # 保存迭代点位置

    # pytorch 自动求导不熟练。grad这种可以求出一个梯度向量的函数，其输入grad(y, x)中，并不表示当位置为x时的梯度
    # 而y.backward()这种，x.data只要发生变化，y重新backward，则x.grad也会改变，但是grad(y, x)只要不重新求y
    # 不管x.data如何变，y记录的都是原来的x位置
    def solve(self, initial, criteria = 1e-3):
        x = Var(initial, requires_grad = True)
        y = self.func(x)
        ndim = x.size()[0]
        p = - grad(y, x)[0]       # 初始化
        last_g = - p.clone()
        i = 0
        while last_g.norm() >= criteria:
            self.pos.append(x.data.numpy())
            print("Iter: %d, (x, y) = (%f, %f)"%(i, x.data[0], x.data[1]))
            x.data = fibonacci(self.func, p.numpy(), x.data.numpy(), max_iter = self.max_ls_iter)
            y = self.func(x)
            g = grad(y, x)[0]
            lamb = g.T @ (g - last_g) / (last_g.T @ last_g) # PRP（不依赖于完全的精确搜索（因为达不到））
            p = - g + lamb * p
            last_g = g.clone()
            i += 1
            if i >= ndim:   # 根据共轭梯度法，n次内收敛。假设不收敛需要重新开始一次
                p = - g
                last_g = - p.clone()
                i = 0
        return self.pos

if __name__ == '__main__':
    conj = Conj(func_to_eval, 2, 16)
    pos = conj.solve(5 * torch.ones(2) + torch.Tensor(np.random.normal(0, 3, 2)))
    draw2D(func_to_eval, pos)