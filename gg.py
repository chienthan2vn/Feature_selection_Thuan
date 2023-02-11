import numpy as np
def boundary(x, lb):
    if x < lb:
        x = lb
    
    return x

a = np.array([[1,2,3,4,5]])
b = np.array([[3,3,3,3,3]])
# if a > b:
#     a = b

print(boundary(a[0,:], b[0,:]))
