#-*-coding:utf-8-*-
"""
   拟牛顿法 + Armijo非精确搜索（看一下Amijo是否好用）
   求解非线性优化最有效的算法之一
   为什么好呢？拟牛顿条件对海塞阵进行了很好的近似，使得收敛较快
   并且H能由构造保证是正定的，每次只需要使用公式修正构造的近似矩阵即可
   实现并理解对称秩1（SR1）以及BFGS（秩-2的更新）的方法
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3
from torch.autograd import Variable as Var
from torch.autograd import grad
from lineSearch import Armijo, rosenBrock, draw2D

def QuasiNewton(func, initial, method = 'sr1', max_iter = 16, criteria = 1e-4):
    x = Var(initial, requires_grad = True).reshape(-1, 1)   # 列向量构造
    pos = [x.data.clone().numpy()]
    # 求二阶导的方法在我看来非常的暴力
    y = func(x)
    g = grad(y, x, create_graph = True)[0]
    H = torch.Tensor([])
    for gval in g:      # 只求一直Hesse阵（小规模问题可以）
        g2 = grad(gval, x, retain_graph = True)[0]
        H = torch.hstack((H, g2))
    print(H)
    g = g.detach()
    x_old = x.data.clone()
    g_old = g.data.clone()
    for i in range(max_iter):
        d = - H @ g
        t = Armijo(func, x, d)
        x.data += t * d / d.norm()
        pos.append(x.data.clone().numpy())
        s = x.data - x_old
        y = func(x)
        g = grad(y, x)[0]
        # print(g)
        if g.norm() < criteria:
            print("Convergence. Exiting...")
            breakpoint
        p = g - g_old
        x_old = x.data.clone()
        g_old = g.clone()
        temp = s - H @ p
        H += temp @ temp.T / (temp.T @ p)
        print("Iter %d, y = %f"%(i, y))
    return x.data.numpy(), pos

if __name__ == "__main__":
    xs, pos = QuasiNewton(rosenBrock, torch.FloatTensor([2, 2]), max_iter = 256)
    draw2D(rosenBrock, pos)
        







