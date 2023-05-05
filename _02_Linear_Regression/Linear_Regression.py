# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge_regression(x, y, alpha):
    """
    计算岭回归的参数

    参数：
    X - 输入特征数据，形状为 (m, n)，其中 m 是样本数，n 是特征数
    y - 输出标签数据，形状为 (m, 1)
    alpha - 正则化强度，一个标量

    返回：
    w - 模型的权重参数，形状为 (n, 1)
    """

    # 计算正则化矩阵
    I = np.identity(x.shape[1])
    A = np.dot(x.T, x) + alpha * I

    # 计算权重参数
    w = np.dot(np.dot(np.linalg.inv(A), x.T), y)

    return w


def lasso_regression(x, y, alpha=0.1, max_iter=100000, lt=1e-12):
    """
    使用Lasso回归对数据进行拟合，返回权重向量 w。
    """

    min = 1e10
    _, n = x.shape
    w = np.zeros(n)
    for i in range(max_iter):
        mse = np.sum(((x @ w) - y.T) @ ((x @ w) - y.T).T)/(np.shape(x)[0])
        l1 = alpha * (np.sum(np.abs(w)))
        lasso_loss = mse + l1
        dw = x.T @ ((x @ w) - y.T) + alpha * np.sign(w)
        w = w - lt * dw
        if np.abs(min - lasso_loss) < 0.0001:
            break
        if min >= lasso_loss:
            min = lasso_loss
            best = w
    return best


def ridge(data):
    x, y = read_data()
    alpha = 1e-12
    weight = ridge_regression(x,y,alpha)
    y_pred = np.dot(data, weight)
    return y_pred


def lasso(data):

    x, y = read_data()
    w = lasso_regression(x,y)
    return data@w[:6] + w[-1]


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

