#-*-coding:utf-8-*-
"""
    Gauss Newton
    Method specialized for LS problems.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3
from torch.autograd import Variable as Var
from torch.autograd import grad

# One specified non-linear function.
def testFunc(dot, param):
    return dot.T @ dot * param[0] + dot[0] * param[1] + dot[1] * param[2] + 1 / (dot.T @ dot + param[3])

"""
    Gauss-Newton Method
    Easy to understand. Knowing that it's costy to compute Hessian Matrix.
    Why don't we just approximate it with Jacobian?
    This Gauss-Newton Method can be applied to LS problem only? Maybe, for in LSP
    The Hessian of error term is $2(J^TJ+S)$, where S can be ignored.

    The problem of 3D curve fitting.
    Each row of data represents a 2D dot(x, y)(for 2D is easy to visualize)
    whereas each row possesses a 3-column structure, for the pos[2] is the oberseved value
"""
def GaussNewton(data, param_init, func, max_iter = 30, criteria = 1e-4):
    def evaluate(dot, param, func):
        val = func(dot[:2], param)
        res = float(val.data - dot[2])
        g = grad(val, param)[0]
        return g, res
    if not type(data) == torch.Tensor:
        data = torch.Tensor(data)
    ndim = len(param_init)
    param = Var(param_init, requires_grad = True)
    for i in range(max_iter):
        residual = torch.zeros((data.size()[0], 1))
        J, residual[0] = evaluate(data[0], param, func)
        for i, dot in enumerate(data[1:]):  # for each sample dot, Jacobian and residual should be evaluated.
            _J, _res = evaluate(dot, param, func)
            J = torch.vstack((J, _J))
            residual[i + 1] = _res
        H_aprx = J.T @ J
        invH, _ =  torch.solve(torch.eye(ndim), H_aprx)
        temp = invH @ (J.T @ residual)
        param.data -= temp.view(-1)
    return param.data.numpy()

def generateData(func, param, xy = -6, n = 20, sigma = 4):
    xx, yy = np.meshgrid(np.linspace(-xy, xy, n), np.linspace(-xy, xy, n))
    dots = np.c_[xx.ravel(), yy.ravel()]
    val = np.array([func(dot, param) for dot in dots])
    val_perturb = val + np.random.normal(0, sigma, val.size)
    truth = np.hstack((dots, val.reshape(-1, 1)))
    noised = np.hstack((dots, val_perturb.reshape(-1, 1)))
    return truth, noised, xx, yy

def showResult(func, param, truth, noised, xx, yy):
    fig = plt.figure()
    ax = mp3.Axes3D(fig)
    xs = truth[:, 0]
    ys = truth[:, 1]
    # ax.plot3D(xs, ys, truth[:, 2], c = 'b')
    ax.scatter3D(xs, ys, truth[:, 2], c = 'b', s = 7)
    ax.scatter3D(noised[:, 0], noised[:, 1], noised[:, 2], c = 'r', s = 7)
    ax.plot_surface(xx, yy, truth[:, 2].reshape(xx.shape), color = 'b', alpha = 0.4)
    dots = truth[:, :2]
    res = np.array([func(dot, param) for dot in dots])
    ax.plot_surface(xx, yy, res.reshape(xx.shape), color = 'g', alpha = 0.4)

    ax.scatter3D(xs, ys, res, c = 'g', s = 7)
    plt.show()

if __name__ == '__main__':
    params = torch.FloatTensor([-0.1, 2, 1, 1])
    truth, noised, xx, yy = generateData(testFunc, params)

    res_param = GaussNewton(torch.Tensor(noised), params, testFunc)
    showResult(testFunc, res_param, truth, noised, xx, yy)