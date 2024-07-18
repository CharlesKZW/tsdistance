import numpy as np
from numba import jit
import math


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
def make_envelopes(X, w): # used to compute lower and upper envelopes
    num_columns = len(X)
    upper_envelopes = np.zeros(num_columns)
    lower_envelopes = np.zeros(num_columns)
    
    for j in range(num_columns):
        wmin = max(0, j-w)
        wmax = min(num_columns-1, j+w)

        upper_envelopes[j] = max(X[wmin: wmax+1])
        lower_envelopes[j] = min(X[wmin: wmax+1])
            
    return upper_envelopes, lower_envelopes

def dev(X):
    lenx = X.shape[1]
    dX = (2 * X[:, 1:lenx-1] + X[:, 2:lenx] - 3*X[:, 0:lenx-2])/4
    first_col = np.array([dX[:, 0]])
    last_col = np.array([dX[:, dX.shape[1]-1]])
    dX = np.concatenate((first_col.T, dX), axis = 1)
    dX = np.concatenate((dX, last_col.T), axis =1)
    return dX

@jit(nopython = True)
def lcss_subcost(x, y, epsilon):
    if abs(x-y) <= epsilon: 
        r = 1
    else:
        r = 0
    return r

def lower_b_n(t,w):
  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = min(t[max(0,i-w):min(len(t)-1,i+w)+1])
  return b


@jit(nopython = True)
def lower_b(t,w):
  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = min(t[max(0,i-w):min(len(t)-1,i+w)+1])
  return b


@jit(nopython = True)
def swale_subcost(x, y, epsilon, p , r):
    if abs(x-y) <= epsilon:
        cost = r
    else:
        cost = p

    return cost

@jit(nopython=True)
def msm_dist(new, x, y, c):
    if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
        dist = c
    else:
        dist = c + min(abs(new - x), abs(new - y))
    return dist

@jit(nopython = True)
def edr_subcost(x, y, epsilon):
    if abs(x-y) <= epsilon:
        cost = 0
    else:
        cost = 1
    return cost