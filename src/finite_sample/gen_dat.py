import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(formatter={'float_kind': '{:.4f}'.format})


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def scm(n, d, ep1):
    A1 = np.random.randn(n, d) * ep1
    A2 = np.random.randn(n, d) * ep1
    A = np.concatenate([A1, A2], 1)
    X = f_x(A)
    T, w, p_proto, X_t = f_t(X)
    Y, ITE = f_y(X, T, w, p_proto, ep1, X_t)
    Z = f_z(T, Y, A)
    unadjust = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    return {"A": A,
            "X": X,
            "Z": Z,
            "T": T,
            "Y": Y,
            'pre_treatment': X_t,
            "ITE": ITE,
            "unadjusted": unadjust,
            'P': sigmoid(p_proto)
            }

def f_x(A):
    np.random.seed(0)
    dim = A.shape[1]
    A_x = A[:, int(dim / 5):int(4 * dim / 5)]
    n, d = A_x.shape
    w_linear = np.random.randn(d, d) / d
    X = A_x @ w_linear + np.random.randn(n, d)
    return X


def f_t(X):
    np.random.seed(1)
    dim = X.shape[1]
    X_t = X[:, :int(dim * 2 / 5)]
    n, d = X_t.shape
    w_linear = np.random.randn(d, d) / d
    X_inter = np.concatenate(
        [X_t[:, :1] * X_t[:, 1:2], X_t[:, 1:2] * X_t[:, 2:4], X_t[:, 2:3] * X_t[:, 3:d] / np.square(X_t).mean()], 1)
    w_inter = np.random.randn(d, d) / d
    p_proto = np.sum(X_t @ w_linear + X_inter @ w_inter, 1).reshape(-1, 1)
    t = np.random.binomial(1, sigmoid(p_proto))

    return t.reshape(-1, 1), (np.sum(w_linear, 1)).reshape(-1, 1), p_proto, X_t


def f_y(X, T, alpha, p, e1, X_t):
    np.random.seed(2)
    n, dim = X.shape

    X_y = X[:, int(dim * 2 / 5):]
    n, d = X_y.shape
    w_linear = np.random.normal(loc=0.5 * alpha, scale=abs(0.5 * alpha))
    w_l2 = np.random.randn(d, 1) / d

    X_inter = np.concatenate(
        [X_y[:, :1] * X_y[:, 4:5], X_y[:, 1:2] * X_y[:, 3:4], X_y[:, 1:2] * X_y[:, 2:d] / np.square(X_y).mean()], 1)
    w_inter = np.random.randn(d, d) / d

    beta = X_t @ w_linear + X_y @ w_l2 + np.sum(X_inter @ w_inter, 1).reshape(-1, 1) + 2 * p

    Y = np.random.binomial(1, sigmoid(1.25 * T + beta))
    ITE = sigmoid(1.25 * np.ones_like(T) + beta).reshape(-1, 1) - sigmoid(beta).reshape(-1, 1)

    print("att here is: ", ITE[T == 1].mean())
    print("naive ate is: ", Y[T == 1].mean() - Y[T == 0].mean())
    return Y.reshape(-1, 1), ITE.reshape(-1, 1)


def f_z(T, Y, A):
    np.random.seed(0)
    n, d = A.shape
    Z = Y + A + np.random.randn(n, d)
    return Z


def draw_samples(ep1, i, n=900, d=25):
    ls = scm(n, d, ep1)
    for num, output in enumerate([ls]):
        np.savez_compressed("../../dat/exp3/replication_{}_ep_{}".format(i, ep1), **output)


if __name__ == '__main__':
    for i in range(10):
        np.random.seed(i)
        draw_samples(0.2, i)
        draw_samples(1, i)
        draw_samples(5, i)
