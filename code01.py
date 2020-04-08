import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


# 目标函数
def real_func(x):
    return np.sin(2 * np.pi * x)

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
# 加上正态分布噪音的目标函数值
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]

def fitting(M=0):
    """M为多项式的次数"""
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print('Fitting Parametes:', p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='nosie')
    plt.legend()
    return p_lsq

# p_lsq_0 = fitting(M=0)
# plt.show()
# p_lsq_1 = fitting(M=1)
# plt.show()
# p_lsq_2 = fitting(M=2)
# plt.show()
# p_lsq_3 = fitting(M=3)
# plt.show()
# p_lsq_4 = fitting(M=4)
# plt.show()
p_lsq_9 = fitting(M=9)
plt.show()
regularization = 0.0001


def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    res = np.sqrt(0.5 * regularization * np.square(p))
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))

    return ret

# 最小二乘法,加正则化项
p_init = np.random.rand(9 + 1)
p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y))
plt.plot(x_points, real_func(x_points), label='real')
plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label='fitted curve')
plt.plot(x_points, fit_func(p_lsq_regularization[0], x_points), label='regularization')
plt.plot(x, y, 'bo', label='nosie')
plt.legend()

plt.show()





