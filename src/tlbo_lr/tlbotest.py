import numpy as np
from numpy.random import rand, randint
import sys
sys.path.append("./src")
from tlbo_lr.fun import fun
import math

"""
Tạo ngẫu nhiên các cá thể cho quần thể đầu tiên với các tham số:
    N:      số lượng cá thể
    dim:    số chiều của cá thể
    lb:     cận dưới = 0
    ub:     cận trên = 1

Input:  lb, ub, N, dim
Output: ma trận 2 chiều X mô tả về các cá thể và tính năng của chúng
"""
def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X

"""
Chuyển đổi các giá trị thập phân trong vùng [0 1] của các tính năng thành
các giá trị nhị phân 0 hoặc 1 đại diện cho tính năng đó có được chọn hay 
không so với các tính năng ban đầu của dữ liệu với các tham số:
    X:      Ma trận 2 chiều mô tả tính năng của các cá thể
    thres:  Ngưỡng convert, > thres đưa về 1, < thres đưa về 0
    N:      Số lượng cá thể
    dim:    Số chiều của cá thể

Input:  X, thres, N, dim
Output: Ma trận Xbin mô tả tính năng của các cá thể sau khi đã convert
sang dạng nhị phân 0 và 1. Trong đó 1 là chọn, 0 là không chọn các tính
năng so với tổng các tính năng gốc 
"""
def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin

"""
Convert các giá trị tính năng khi chúng vượt quá 1 hoặc 0 sau khi đi qua
thuật toán. Nếu nhỏ hơn 0 đưa về 0, lớn hơn 1 đưa về 1. Các tham số bao gồm:
    x:  Giá trị cần convert khi vượt quá ngưỡng
    lb: Cận dưới = 0
    ub: Cận trên = 1

Input:  x, lb, ub
Output: Số x sau khi đã convert về trong ngưỡng
"""
def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x

"""
Thuật toán TLBO
"""
def jfs(xtrain, ytrain, opts):
    # Parameter
    ub = 1
    lb = 0
    thres = 0.5
    imbalanced = 0
    r = rand(1) #0 to 1

    # opts
    N           = opts['N']
    max_iter    = opts['max_iter']
    if 'r' in opts:
        r = opts['r']
    if 'imbalanced' in opts:
        imbalanced = opts['imbalanced']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin  = binary_conversion(X, thres, N, dim)

    """
    fit     : ma trận hàng của hàm mục tiêu cho các cá thể
    Xgb     : Cá thể tốt nhất
    fitG    : Hàm mục tiêu tốt nhất
    """
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')

    # Gia tri fitG tốt nhất qua mỗi vòng lặp
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0

    """
    Vòng lặp thể hiện yêu cầu bài toán
    """
    while t < max_iter:
        """
        Tính toán cá thể tốt nhất cho quần thể với các bước:
        1. Đưa quần thể đã chuyển đổi tính năng về nhị phân (Xbin) để tính toán fitness
        2. Lưu lại cá thể tốt nhất (Xgb) và fitness tốt nhất (fitG)
        """
        for i in range(N):
            fit[i,0] = fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]

        """
        Quá trình Teaching phase: đây là quá trình giúp các cá thể học được
        những điều quan trọng nhất của cá thể giáo viên (fitness nhỏ nhất)
        thông qua công thức x_new = x + r*(x_best - Tf*Xmean). Nếu sự học
        tập được coi là tốt hơn (fitness nhỏ hơn) sẽ thay thế cá thể x thành
        cá thể x_new và cập nhập lại fitness cũng như Xbg và fitG cho vòng lặp

        """
        for i in range(N):
            #Tính toán Xmean
            Xmean = np.sum(X, axis=0)/np.size(X, axis=0)

            #Giá trị Tf là giá trị ngẫu nhiên 1 hoặc 2
            Tf = round(1+rand(1)[0])

            #Tính lại giá trị x 
            Xnew = X[i,:] + r*(Xgb[0,:] - Tf*Xmean)

            # Boudary
            for d in range(dim):
                Xnew[d] = boundary(Xnew[d], lb[0,d], ub[0,d])

            temp = np.zeros([1, dim], dtype='float')
            temp[0,:] = Xnew 
            Xbin = binary_conversion(temp, thres, 1, dim)

            # fitness
            fnew  = fun(xtrain, ytrain, Xbin[0,:], opts)
            
            # update X[i]       
            if fit[i,0] > fnew:
                X[i,:] = Xnew
                fit[i,0] = fnew

            if fitG > fnew:
                Xgb[0,:] = Xnew
                fitG     = fnew

            """
            Quá trình Learning phase: đây là quá trình cho các cá thể
            học tập lẫn nhau, cá thể nào tốt hơn (fitness) sẽ có xu hướng
            được cá thế khác tiến lại gần
            """
            # Lấy cá thể ngẫu nhiên cho quá trình learning phase
            Xparter = np.expand_dims(X[randint(N),:], axis=0)

            # convert về dạng nhị phân để chọn lọc tính năng
            Xparter_bin = binary_conversion(Xparter, thres, 1, dim)

            # Tính toán fitness
            fitparter = fun(xtrain, ytrain, Xparter_bin[0,:], opts)

            # thuật toán so sánh fitness và tính lại x_new
            if(fit[i,0] < fitparter):
                Xnew = X[i,:] + r*(X[i,:] - Xparter[0,:])
            else:
                Xnew = X[i,:] - r*(X[i,:] - Xparter[0,:])

            # Cập nhập lại x_new nếu kết quả (fitness) tốt hơn
            for d in range(dim):
                Xnew[d] = boundary(Xnew[d], lb[0,d], ub[0,d])

            # convert về dạng nhị phân để chọn lọc tính năng
            Xbinnew = binary_conversion(np.expand_dims(Xnew, axis=0), thres, 1, dim)

            # Tính toán fitness cho x_new sau quá trình learning phase
            fitnew = fun(xtrain, ytrain, Xbinnew[0,:], opts)

            # Cập nhập vị trí và fitness nếu quá trình này tốt hơn
            if(fitnew < fit[i,0]):
                X[i,:] = Xnew
                fit[i,0]     = fitnew
            
            # Cập nhập so với Xgb và fitG
            if (fitG > fitnew):
                Xgb[0,:] = Xnew
                fitG     = fitnew

        # Store result
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (FA):", curve[0,t])
        t += 1

    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    tlbo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return tlbo_data    