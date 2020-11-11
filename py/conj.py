#-*-coding:utf-8-*-
"""
    PRP共轭梯度法 + Fibonacci搜索
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable as Var
from torch.autograd import grad

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

def getFibonacci(x):
    return 1 / sqr5 * ((0.5 + 0.5 * sqr5)**x - (0.5 - 0.5 * sqr5)**x)

class Conj:
    # func为pytorch自动求导定义下的函数
    # 线搜索最大迭代次数
    # self.func可能需要实现两套：输入为torch.tensor 与输入为 numpy.ndarray
    def __init__(self, func, dim = 2, max_ls_iter = 16):
        self.func = func
        self.max_ls_iter = max_ls_iter
        self.n = dim
        self.pos = []       # 保存迭代点位置

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
    def __fibonacci__(func, d, x, step_scl = 2, max_range = 32, max_iter = 16, prec = 1e-4):
        dn = np.linalg.norm(d)
        if dn < 1e-6:
            raise ValueError("Possible numeric problem detected. Norm too small.")
        direct = d / dn
        old_val = func(x)
        a = x.copy()
        b = np.zeros_like(x)
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
        return torch.Tensor((a + b) / 2)
    
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
            x.data = Conj.__fibonacci__(self.func, p.numpy(), x.data.numpy(), max_iter = self.max_ls_iter)
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

    # 二维优化问题的可视化
    def draw(self):
        xy = np.array(self.pos)
        xs = xy[:, 0]
        ys = xy[:, 1]
        xlb = min(xs)
        xub = max(xs)
        ylb = min(ys)
        yub = max(ys)
        x, y = self.pos[-1]
        xx, yy = np.meshgrid(np.linspace(min(int(x) - 5, xlb - 1), max(int(x) + 6, xub + 1), 200),
            np.linspace(min(int(y) - 5, ylb - 1), max(int(y) + 6, yub + 1), 200)
        )
        dots = np.c_[xx.ravel(), yy.ravel()]
        dot_num = dots.shape[0]
        res = np.array([self.func(dots[i, :]) for i in range(dot_num)])
        res = res.reshape(xx.shape)
        cmap = plt.get_cmap('bwr')
        plt.contourf(xx, yy, res, cmap = cmap)
        plt.plot(xs, ys, c = 'k')
        plt.scatter(xs, ys, c = 'k', s = 7)
        print(self.pos)
        plt.show()

if __name__ == '__main__':
    conj = Conj(func_to_eval, 2, 16)
    conj.solve(5 * torch.ones(2) + torch.Tensor(np.random.normal(0, 3, 2)))
    conj.draw()