import numpy as np
from sklearn.linear_model import LinearRegression as LR

def LRB_univar_e(x, k):

    assert x.ndim == 1, "x must be a vector"

    X_rm = roll_mat_gen(x, k)

    X_des = x[k:]

    LR_M = LR().fit(X_rm, X_des)

    return 1 - LR_M.score(X_rm, X_des)

def LRB_mulvar_e(x, y, k):

    assert x.ndim == 1 and y.ndim == 1, "x and y must be a vector"

    X_rm = roll_mat_gen(x, k)
    Y_rm = roll_mat_gen(y, k)

    XY = np.concatenate([X_rm.T, Y_rm.T]).T

    X_des = x[k:]

    LR_M = LR().fit(XY, X_des)

    return 1 - LR_M.score(XY, X_des)

def roll_mat_gen(x, k):

    assert x.ndim == 1, "x must be a vector"

    X_rm = []

    N = len(x)

    for i in range(N - k):

        X_rm.append(x[i : i + k])

    return np.array(X_rm)

def univar_AR_est(x_t, order):
    
    length = len(x_t)

    X_mat = np.zeros((length - order, order))
    X_vec = np.zeros((length - order))

    for i in range(length - order):

        X_mat[i, :] = x_t[i : i + order]
        X_vec[i] = x_t[i + order]

    return np.flip(np.linalg.pinv(X_mat) @ X_vec)

def mulvar_AR_est(x_t, y_t, a_order, b_order):
    
    length = len(x_t)

    X_mat = np.zeros((length - a_order, a_order + b_order))
    X_vec = np.zeros((length - a_order))

    for i in range(length - a_order):

        X_mat[i, : a_order] = x_t[i : i + a_order]
        X_mat[i, a_order : a_order + b_order] = y_t[i : i + b_order]
        
        X_vec[i] = x_t[i + a_order]

    coef_est = np.linalg.pinv(X_mat) @ X_vec
    
    a_est = np.flip(coef_est[:a_order])
    b_est = np.flip(coef_est[a_order : a_order + b_order])
    
    return a_est, b_est

def a_estimation_err(a_est, x_t):
    
    order = len(a_est)
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:order] = x_t[:order]

    for i in range(order, length):

        for j in range(order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]

    return np.sum((x_t - x_t_rec) ** 2)

def ab_estimation_err(a_est, b_est, x_t, y_t):
    
    a_order = len(a_est)
    b_order = len(b_est)
    
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:a_order] = x_t[:a_order]

    for i in range(a_order, length):

        for j in range(a_order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]
            
        for j in range(b_order):

            x_t_rec[i] = x_t_rec[i] + b_est[j] * y_t[i - j - 1]

    return np.sum((x_t - x_t_rec) ** 2)