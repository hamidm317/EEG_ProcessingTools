import numpy as np
from Utils import GC

from sklearn.linear_model import LinearRegression as LR

def LRB_GC(x, y, k):

    assert x.ndim == 1 and y.ndim == 1, "x and y must be a vector"

    X_rm = GC.roll_mat_gen(x, k)
    Y_rm = GC.roll_mat_gen(y, k)

    XY = np.concatenate([X_rm.T, Y_rm.T]).T

    X_des = x[k:]

    LR_M = LR().fit(XY, X_des)

    return 1 - LR_M.score(XY, X_des)