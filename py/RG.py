"""
    简约梯度法(Reduced Gradient)
    感觉这个算法实现起来就比较简单了
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.autograd import Variable as Var
from lineSearch import Armijo, wolfeFunc, draw2D

class RG:
    def __init__(self, func, A, b):
        self.func = func
        self.A = A
        self.b = b
        self.m = A.size()[0]     # 约束个数（基变量个数m）
        self.n = A.size()[1]     # 变量个数n

    def getWholeP(self, rs, x, bidx, nidx, invBN):
        p = torch.zeros((self.n, 1))
        pn = torch.zeros_like(rs)
        xn = x[nidx]
        for i, r in enumerate(rs):
            if r <= 0:
                pn[i] = - r
            else:
                pn[i] = - xn[i] * r
        p[nidx] = pn
        p[bidx] = - invBN @ pn
        return p

    def solve(self, initial, max_iter = 256, criteria = 1e-6):
        pos = []
        x = Var(initial.reshape(-1, 1), requires_grad = True)
        for i in range(max_iter):
            _this = x.data[:2].clone()
            pos.append(_this.numpy().ravel())
            idx = torch.argsort(x.view(-1), descending = True)
            bidx = idx[:self.m]
            nidx = idx[self.m:]
            B = self.A[:, bidx]
            N = self.A[:, nidx]
            # ============ 以上为确定 B,N ==============
            y = self.func(x)
            g = grad(y, x)[0]
            gn = g[nidx]
            gb = g[bidx]
            iB, _ = torch.solve(torch.eye(self.m), B)
            temp = iB @ N
            r = gn - temp.T @ gb
            p = self.getWholeP(r, x.data, bidx, nidx, temp)
            # print("X = ", x.data)
            # print("P = ", p)
            if p.norm() < criteria:     # Pn, Pb 为0了
                print("Convergence before max iteration.")
                return x.data.numpy()
            max_t = float("inf")
            t_changed = False
            for j, pv in enumerate(p):
                if pv < 0:      # 存在小于0的p分量才会进行判定
                    rate = x.data[j] / (- pv)
                    if rate < max_t:
                        max_t = rate
                        t_changed = True
            if not t_changed:
                print("Step could be infinite. Minimum is neg-inifinite.")
                return x.data.numpy()
            print("Max t = ", max_t, i)
            t = min(Armijo(self.func, x.data[:2], p[:2]), max_t)
            x.data += t * p
            print(x.view(-1), t)
        return x.data.numpy(), pos

if __name__ == "__main__":
    A = torch.FloatTensor([
        [1, 1, 1, 0, 0, 0],
        [1, 5, 0, 1, 0, 0],
        [-1, 0, 0, 0, 1, 0],
        [0, -1, 0, 0, 0, 1]
    ])
    b = torch.FloatTensor([2, 5, 0, 0]).reshape(-1, 1)
    rg = RG(wolfeFunc, A, b)
    x_best, pos = rg.solve(torch.FloatTensor([0, 0, 2, 5, 1, 1]).reshape(-1, 1))
    draw2D(wolfeFunc, pos)