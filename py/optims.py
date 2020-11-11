#-*-coding:utf-8-*-
"""
    高斯牛顿法
    拟牛顿法
    列文伯格·马尔夸特 信赖域(Trust Region)法
"""
from struct import error
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable as Var
from torch.autograd import grad

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
    def evaluate(data, param, func):
        ndots = data.size()[0]
        error = Var(torch.zeros(0), requires_grad = True)
        residual = torch.zeros(ndots)
        for i in range(ndots):
            x = Var(data[i, :2], requires_grad = False)
            residual[i] = func(x, param) - data[i, 2]
            error += residual[i] ** 2
        return error, residual
    ndim = len(param_init)
    param = Var(param_init, requires_grad = True)
    for i in range(max_iter):
        error, res = evaluate(data, param, func)
        J = grad(error, param)
        if J.norm() < criteria:
            print("Convergence. Jacobian norm less than ", criteria)
            break
        H_aprx = J.T @ J
        invH, _ =  torch.solve(torch.eye(ndim), H_aprx)
        param.data -= invH @ J.T @ res
    return param.data.numpy()


