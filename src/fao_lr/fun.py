from sklearn.linear_model import LogisticRegression
import numpy as np
def fun(x, y, fea, opts, alpha=0.9):
    if np.count_nonzero(fea) == 0: X_subset = x
    else: X_subset = x[:,fea==1]
    lr = LogisticRegression()
    lr.fit(X_subset, y)
    p = (lr.predict(X_subset) == y).mean()
    j = (alpha * (1.0 - p) + (1.0 - alpha) * (1 - (X_subset.shape[1] / x.shape[1])))
    
    return j