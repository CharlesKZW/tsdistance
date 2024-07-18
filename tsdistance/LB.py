#%%
import numpy as np
from numba import jit
from math import floor, sqrt

#%%


@jit(nopython = True)
def lb_keogh(y, x, w):

    XUE, XLE = MakeEnvForSingleTS(x, w)

    leny = len(y)
    lb_dist = 0
    
    for i in range(leny):

        if y[i] > XUE[i]:
            lb_dist += (y[i] - XUE[i]) ** 2
        if y[i] < XLE[i]:
            lb_dist += (y[i] - XLE[i]) ** 2

    return sqrt(lb_dist)

@jit(nopython = True)
def lb_kim(y, x):
    lb_dist = max(abs(x[0] - y[0]),
                  abs(x[len(x)-1] - y[len(y)-1]),
                  abs(max(x)- max(y)),
                  abs(min(x)- min(y)))
    return lb_dist

@jit(nopython = True)
def lb_new(y, x, w):
    leny = len(y)
    lenx = len(x)
    lb_dist = (x[0]-y[0]) ** 2 + (x[lenx-1] - y[leny-1]) **2
    
    for i in range(1,leny-1):

        wmin = max(0, i - w)
        wmax = min(lenx - 1, i + w) 
        wx = np.array([i for i in x[wmin : wmax + 1]])
        Y = np.full(wx.shape[0], -y[i])
        
        diff = np.add(wx, Y)
        cost = min(np.square(diff))

        lb_dist = lb_dist + cost

    return sqrt(lb_dist)

@jit(nopython = True)
def envelope_cost(x, YUE, YLE): # note the returned value is distance squared
    lenx = len(x)
    x_dist = 0

    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    return x_dist

@jit(nopython = True)
def boundary_cost(x, y): # note the returned value is distance squared
    lenx = len(x)
    leny = len(y)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    return fixed_dist


@jit(nopython = True)
def glb_dtw(y, x, w):
    XUE, XLE = MakeEnvForSingleTS(x, w)
    YUE, YLE = MakeEnvForSingleTS(y, w)
    leny = len(y)
    lenx = len(x)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    lb_dist = fixed_dist + max(x_dist, y_dist)


    return sqrt(lb_dist)


@jit(nopython = True)
def glb_dtw_QueryOnly(y, x, XUE, XLE): 
    # lb_keogh vs glb_dtw_QueryOnly: lb_keogh might capture boundary, 
    # but glb_dtw_QueryOnly strictly doesn't.
    leny = len(y)
    lenx = len(x)
    
    fixed_dist = 0

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    

    lb_dist = fixed_dist + max(x_dist, y_dist)


    return sqrt(lb_dist)


@jit(nopython = True)
def glb_dtw_QueryBoundary(y, x, XUE, XLE):
    leny = len(y)
    lenx = len(x)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2
    

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    

    lb_dist = fixed_dist + max(x_dist, y_dist)


    return sqrt(lb_dist)

@jit(nopython = True)
def glb_dtw_QueryData(y, x, XUE, XLE, YUE, YLE):
    leny = len(y)
    lenx = len(x)
    
    
    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    lb_dist = max(x_dist, y_dist)

    return sqrt(lb_dist)


@jit(nopython = True)
def lower_b(t,w):

  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = min(t[max(0,i-w):min(len(t)-1,i+w)+1])
  
  return b

@jit(nopython = True)
def upper_b(t,w):

  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = max(t[max(0,i-w):min(len(t)-1,i+w)+1])
  
  return b
    
@jit(nopython = True)
def lb_keogh_squared(x,u,l):
  sumd = 0
  for i in range(len(x)):
    if x[i] > u[i]:
      sumd += (x[i] - u[i]) ** 2
    if x[i] < l[i]:
      sumd += (x[i] - l[i]) ** 2
    
  return sumd 

@jit(nopython = True)
def lb_improved(x,y,w, YUE, YLE):
    h = []
    l = YLE
    u = YUE
    for i in range(len(y)):
        if x[i] <= l[i]:
            h.append(l[i])
        elif x[i] >= u[i]:
            h.append(u[i])
        else:
            h.append(x[i])

    upper_h = upper_b(h,w)
    lower_h = lower_b(h,w)

    return sqrt(lb_keogh_squared(x,u,l) + lb_keogh_squared(y,upper_h,lower_h))

@jit(nopython = True)
def lb_wdtw_A(y, x, g, XUE, XLE):
    
    leny = len(y)
    lenx = len(x)
    lb_dist = 0
    

    for i in range(leny):

        if y[i] > XUE[i]:
            lb_dist += (y[i] - XUE[i]) ** 2
        if y[i] < XLE[i]:
            lb_dist += (y[i] - XLE[i]) ** 2

    w0 = min([1 / (1 + np.exp(-g * (i - len(x) / 2))) for i in
                         range(0, len(x))])
    lb_dist = lb_dist * w0
    return sqrt(lb_dist)

@jit(nopython = True)
def lb_wdtw_B(y, x, g, XUE, XLE, YUE, YLE):
    leny = len(y)
    lenx = len(x)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    lb_dist = fixed_dist + max(x_dist, y_dist)

    w0 = min([1 / (1 + np.exp(-g * (i - len(x) / 2))) for i in
                         range(0, len(x))])
    
    lb_dist = lb_dist * w0
    return sqrt(lb_dist)


def dev(X):
    lenx = X.shape[1]
    dX = (2 * X[:, 1:lenx-1] + X[:, 2:lenx] - 3*X[:, 0:lenx-2])/4
    first_col = np.array([dX[:, 0]])
    last_col = np.array([dX[:, dX.shape[1]-1]])
    dX = np.concatenate((first_col.T, dX), axis = 1)
    dX = np.concatenate((dX, last_col.T), axis =1)
    return dX

@jit(nopython = True)
def lb_ddtw(y, x, XUE, XLE, YUE, YLE):

    leny = len(y)
    lenx = len(x)

    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    lb_dist = fixed_dist + max(x_dist, y_dist)

    return sqrt(lb_dist)


@jit(nopython = True)
def lb_keogh_lcss(y, x,epsilon, XUE, XLE): # LB_Keogh_LCSS
    
    LE_lower = np.subtract(XLE,epsilon)
    UE_higher = np.add(XUE,epsilon)
    
    leny = len(y)
    
    sum = 0
    
    for i in range(leny):

        if y[i] >= LE_lower[i] and y[i] <= UE_higher[i]:
            sum += 1

    lb_dist = 1 - (sum/(min(len(x),len(y))))

    return lb_dist

@jit(nopython = True)
def lcss_subcost(x, y, epsilon):
    if abs(x-y) <= epsilon: 
        r = 1
    else:
        r = 0
    return r

@jit(nopython = True)
def glb_lcss(y, x, epsilon, XUE, XLE, YUE, YLE):
    lenx = len(x)
    leny = len(y)
    fixed_sum = lcss_subcost(x[0], y[0], epsilon) + lcss_subcost(x[lenx-1], y[leny-1], epsilon)
    
    XLE_lower = np.subtract(XLE,epsilon)
    XUE_higher = np.add(XUE,epsilon)
    
    y_reward = 0
    
    for i in range(1, leny-1):
    
        if y[i] >= XLE_lower[i] and y[i] <= XUE_higher[i]:
            y_reward += 1
    
    YLE_lower = np.subtract(YLE,epsilon)
    YUE_higher = np.add(YUE,epsilon)
    x_reward = 0
   
    for i in range(1, lenx-1):
       
        if x[i] >= YLE_lower[i] and x[i] <= YUE_higher[i]:
            x_reward += 1
    
    sum = fixed_sum + min(y_reward, x_reward)
    lb_dist = 1 - (sum/(min(len(x),len(y))))
    
    return lb_dist


@jit(nopython = True)
def lb_erp(x, y): 
    return abs(np.sum(x) - np.sum(y))

@jit(nopython = True)
def lb_kim_erp(y, x, m): 
    x_max = max(m, max(x))
    x_min = min(m, min(x))
    lb_dist = max(abs(x[0]-y[0]),
                  abs(x[len(x)-1] - y[len(y)-1]),
                  abs(x_max - max(y)),
                  abs(x_min - min(y)))
    return lb_dist

@jit(nopython = True)
def lb_keogh_erp(y, x, m, XUE, XLE): 
    leny = len(y)
    lenx = len(x)
    lb_dist = 0
    for i in range(leny):
        UE = max(m, XUE[i])
        LE = min(m, XLE[i])
        if y[i] > UE:
            lb_dist += (y[i] - UE) ** 2
        if y[i] < LE:
            lb_dist += (y[i] - LE) ** 2
    return sqrt(lb_dist)


@jit(nopython = True)
def glb_erp(y, x, m, XUE, XLE, YUE, YLE): # GLB_ERP
    lenx = len(x)
    leny = len(y)
    
    fixed_dist = min((x[lenx-1] - y[leny-1])**2, (x[lenx-1]-m)**2, (y[leny-1]- m)**2)

    y_dist = 0
    for i in range(1, leny-1):

    
        if y[i] > XUE[i]:
            y_dist += min((y[i]-XUE[i])**2, (y[i]-m)**2)
        elif y[i] < XLE[i]:
            y_dist += min((y[i]-XLE[i])**2, (y[i]-m)**2)
    
    x_dist = 0
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += min((x[i]-YUE[i])**2, (x[i]-m)**2)
        elif x[i] < YLE[i]:
            x_dist += min((x[i]-YLE[i])**2, (x[i]-m)**2)

    lb_dist = fixed_dist + max(x_dist, y_dist)
    
    return sqrt(lb_dist)



@jit(nopython = True)
def edr_subcost(x, y, epsilon):
    if abs(x-y) <= epsilon:
        cost = 0
    else:
        cost = 1
    return cost

@jit(nopython = True)
def glb_edr(x, y, epsilon, XUE, XLE, YUE, YLE):
    lenx = len(x)
    leny = len(y)
    fixed_cost = 0 + min(edr_subcost(x[lenx-1], y[leny-1], epsilon), 1)
    y_dist = 0
    for i in range(1, leny-1):
        if y[i] > XUE[i]:
            y_dist += edr_subcost(y[i], XUE[i], epsilon)
        if y[i] < XLE[i]:
            y_dist += edr_subcost(y[i], XLE[i], epsilon)
    x_dist = 0
    for i in range(1, lenx-1):
        if x[i] > YUE[i]:
            x_dist += edr_subcost(x[i], YUE[i], epsilon)
        if x[i] < YLE[i]:
            x_dist += edr_subcost(x[i], YLE[i], epsilon)

    lb_dist = fixed_cost + max(x_dist, y_dist)

    return lb_dist




@jit(nopython = True)
def lb_msm(y, x, c, XUE, XLE):

    lenx = len(x)
    leny = len(y)

    lb_dist = abs(x[0]-y[0])

    for i in range(1,leny):

        if y[i] > XUE[i] and y[i-1] >= y[i]:
            lb_dist += min(abs(y[i]-XUE[i]), c)
        if y[i] < XLE[i] and y[i-1] <= y[i]:
            lb_dist += min(abs(y[i]-XLE[i]), c)
    
    return lb_dist




@jit(nopython = True)
def lb_msm_C(x, y, c, w):
    lenx = len(x)
    leny = len(y)

    lb_dist = abs(x[0]-y[0])

    for i in range(1,leny):

        wmin = max(1, i - w)
        wmax = min(lenx - 1, i + w) 

        UE = max(x[wmin : wmax + 1])
        LE = min(x[wmin : wmax + 1])

        if y[i] > UE and y[i-1] >= y[i]:
            lb_dist += min(abs(y[i]-UE), c)
        if y[i] < LE and y[i-1] <= y[i]:
            lb_dist += min(abs(y[i]-LE), c)
        if y[i] > max(UE, y[i-1]):
            lb_dist += min(abs(y[i]-UE), c+abs(y[i]-y[i-1]))
        if y[i] < min(LE, y[i-1]):
            lb_dist += min(abs(y[i]-LE), c+abs(y[i]-y[i-1]))
    
    return lb_dist


@jit(nopython = True)
def lb_twed(y, x, lamb, nu, XUE, XLE):
    leny = len(y)

    lb_dist = min((x[0] - y[0])**2, (x[0])**2 + nu + lamb, (y[0])**2 + nu + lamb)

    for i in range(1, leny):

        if y[i] > max(XUE[i], y[i-1]):
            lb_dist += min(nu, (y[i]- max(XUE[i], y[i-1]))**2)
        if y[i] < min(XLE[i], y[i-1]):
            lb_dist += min(nu, (y[i]- min(XLE[i], y[i-1]))**2)
        
    return lb_dist


@jit(nopython = True)
def glb_twed(x, y, lamb, XUE, XLE, YUE, YLE):
    leny = len(y)
    lenx = len(x)

    fixed_dist = abs(x[0]-y[0]) + min(
                                    abs(x[lenx-1]-y[leny-1]),
                                    abs(y[lenx-1]-y[lenx-2])+lamb,
                                    abs(x[lenx-1]-x[lenx-2])+lamb
                                    )

    
    y_dist = 0
    for i in range(1, leny-1):

        if y[i]>=XUE[i] and y[i-1]>=XUE[i]:
            y_dist += min((abs(y[i]-XUE[i]) + abs(y[i-1]-XUE[i])), (abs(y[i]-y[i-1])+lamb))
        if y[i]<=XLE[i] and y[i-1]<=XLE[i]:
            y_dist += min(abs(y[i]-XLE[i]) + abs(y[i-1]-XLE[i]), abs(y[i]-y[i-1])+lamb)

    x_dist = 0
    for i in range(1, lenx-1):

        if x[i]>=YUE[i] and x[i-1]>=YUE[i]:
            x_dist += min((abs(x[i]-YUE[i]) + abs(x[i-1]-YUE[i])), (abs(x[i]-y[i-1])+lamb))
        if y[i]<=YLE[i] and y[i-1]<=YLE[i]:
            x_dist += min(abs(x[i]-YLE[i]) + abs(x[i-1]-YLE[i]), abs(x[i]-x[i-1])+lamb)

    lb_dist = fixed_dist + max(y_dist, x_dist)

    return lb_dist


@jit(nopython = True)
def matlab_round(value):
    rounded_value = round(value)
    if value - rounded_value == 0.5:
        rounded_value +=1
    return rounded_value

@jit(nopython = True)  
def MakeEnvForSingleTS(ts, w):
    ts_length = len(ts)
    UE = np.zeros(ts_length)
    LE = np.zeros(ts_length)

    for j in range(ts_length):
            wmin = max(0, j-w)
            wmax = min(ts_length-1, j+w)

            UE[j] = max(ts[wmin: wmax+1])
            LE[j] = min(ts[wmin: wmax+1])
    
    return UE, LE


@jit(nopython = True)     
def make_envelopes(X, w):
    num_rows = X.shape[0]
    num_columns = X.shape[1]
    upper_envelopes = np.zeros((num_rows, num_columns))
    lower_envelopes = np.zeros((num_rows, num_columns))
    

    for i in range(num_rows):
        for j in range(num_columns):
            wmin = max(0, j-w)
            wmax = min(num_columns-1, j+w)

            upper_envelopes[i, j] = max(X[i, wmin: wmax+1])
            lower_envelopes[i, j] = min(X[i, wmin: wmax+1])
            

    return upper_envelopes, lower_envelopes

@jit(nopython = True) 
def add_projection(x, y, YUE, YLE, keogh, window):
    H = []
    for i in range(len(y)):
        if x[i] <= YLE[i]:
            H.append(YLE[i])
        elif x[i] >= YUE[i]:
            H.append(YUE[i])
        else:
            H.append(x[i])
    
    HUE = upper_b(H, window)
    HLE = lower_b(H, window)

    lb_dist = np.sqrt(keogh **2 + lb_keogh_squared(y, HUE, HLE))

    return lb_dist
        

