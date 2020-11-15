#-*-coding:utf-8-*-
"""
    各种线搜索方法 / 测试函数
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.autograd import Variable as Var

sqr5 = np.sqrt(5)

def getFibonacci(x):
    return 1 / sqr5 * ((0.5 + 0.5 * sqr5)**x - (0.5 - 0.5 * sqr5)**x)

"""
    func 为待估计函数
    d 为当前方向
    x 为初始点
    step_scl(scale) 步长
    max_iter 最大搜索次数
    返回一个步长值
    一般我们不会输入一个过小的d，否则在求单位向量的时候可能出现数值问题
"""
def fibonacci(func, d, x, step_scl = 2, max_range = 32, max_iter = 16, prec = 1e-4):
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

def Armijo(func, x, p, m = 0.2, M = 4, max_iter = 16):
    direct = p / p.norm()
    t = Var(torch.zeros(1), requires_grad = True)
    y = func(x + t * direct)
    g = grad(y, t)[0]              # 求的是一维导数：对于t的导数
    a = x.clone()
    b = a - 0.5 * func(a) / (g * m)   # 开始时对t = 0处求导 （选择开始点1/2函数值的位置）
    # (0.5 *f(a) - f(a)) / (b - a) = m * f'(a)
    # 但个人觉得这种方式不好，这种方式对于初始点>0,最值点<0的情况不能有很好的处理
    threshold = func(x) + m * g
    t.data = (b - a).norm()     # direct为单位向量
    for i in range(max_iter):
        val = func(x + t * direct)
        if val > threshold:
            t.data *= 0.5 * (1 + 1 / M)            # 最后得到的是一个满意解 从这种意义上说，GoldStein感觉更好一些？
        else:
            break
    return t.data

# RosenBrock 香蕉♂函数
def rosenBrock(x, a = 10):
    return (1 - x[0]) ** 2 + a * (x[1] - x[0] ** 2) ** 2

# 《运筹学》 书 24题
def wolfeFunc(x):
    return 2 * x[0] ** 2 + 2 * x[1] ** 2 - 2 * x[0] * x[1] - 4 * x[0] - 6 * x[1] + 6

def draw2D(func, pos):
    xy = np.array(pos)
    # print(xy)
    xs = xy[:, 0]
    ys = xy[:, 1]
    xlb = min(xs)
    xub = max(xs)
    ylb = min(ys)
    yub = max(ys)
    x, y = pos[-1]
    x_range = np.linspace(min(int(x) - 5, xlb - 1), max(int(x) + 6, xub + 1), 200)
    xx, yy = np.meshgrid(x_range, np.linspace(min(int(y) - 5, ylb - 1), max(int(y) + 6, yub + 1), 200))
    dots = np.c_[xx.ravel(), yy.ravel()]
    dot_num = dots.shape[0]
    res = np.array([func(dots[i, :]) for i in range(dot_num)])
    res = res.reshape(xx.shape)
    cmap = plt.get_cmap('bwr')
    plt.contourf(xx, yy, res, cmap = cmap)
    plt.plot(xs, ys, c = 'k')
    plt.scatter(xs, ys, c = 'k', s = 7)
    # cys = (5 - x_range) / 5
    # plt.plot(x_range, cys, c = 'r')
    plt.show()


