import numpy as np
from numba import jit
import math


# a helper function for ddtw, dlcss, lb_ddtw
def dev(X):
    lenx = X.shape[1]
    dX = (2 * X[:, 1:lenx-1] + X[:, 2:lenx] - 3*X[:, 0:lenx-2])/4
    first_col = np.array([dX[:, 0]])
    last_col = np.array([dX[:, dX.shape[1]-1]])
    dX = np.concatenate((first_col.T, dX), axis = 1)
    dX = np.concatenate((dX, last_col.T), axis =1)
    return dX

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

# Start of DTW
def dtw(x, y, w = None, constraint=None, fast = True):

    r"""Dynamic Time Warping (DTW) [1]_ utilizes dynamic programming to find 
    the optimal alignment between elements of times series :math:`X = (x_{1}, x_{2}, ..., x_{n})` 
    and :math:`Y = (y_{1}, y_{2}, ..., y_m)` by constructing 
    a distance matrix :math:`M` of shape :math:`(n, m)` with the following forumla:

    .. math::

        M_{i, j} = \begin{cases}
            0, \ i = j = 0 \\
            d_{x_i, y_j} + min \begin{cases}
                M_{i-1, j}\\
                M_{i, j-1} \\
                M_{i-1, j-1}\\
                \end{cases} where \ d_{x_i, y_j}=|x_i - y_j|
        \end{cases}
    
    and return  the DTW distance :math:`M_{n, m}`.

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series. ``w`` defaults to ``None``, in which case the warping window is the length of the longest time series.
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``
    :type fast: bool, optional
    :return: DTW distance
    

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.ElasticMeasures import dtw
        import numpy as np

        X = np.array([3, 2, 4, 5, 5, 2, 4, 7, 9, 8])
        Y = np.array([3, 3, 3, 1, 6, 9, 9])

        dtw_dist = dtw(X, Y, 'Sakoe-Chiba', 3)
        dtw_dist

    Output:

    .. code-block:: bash

      4.123105625617661

    **References**

    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """

    if constraint == "None":
        if fast == True:
            return dtw_n_numba(x, y)
        if fast == False:
            return dtw_n(x, y)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return dtw_scb_numba(x, y, w)
        if fast == False:
            return dtw_scb(x, y, w)
    
def dtw_n(x, y):
    N = len(x)
    M = len(y)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(1, M):
            cost = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

    Dist = math.sqrt(D[N, M])

    return Dist

def dtw_scb(x, y, w):
    N = len(x)
    M = len(y)
    if w == None:
        w = max(N, M)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(max(1, i-w), min(i+w, M)+1):
            cost = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

    Dist = math.sqrt(D[N, M])

    return Dist

@jit(nopython=True)
def dtw_n_numba(x, y):
    N = len(x)
    M = len(y)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(1, M):
            cost = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

    Dist = math.sqrt(D[N, M])

    return Dist

@jit(nopython=True)
def dtw_scb_numba(x, y, w):
    N = len(x)
    M = len(y)
    if w == None:
        w = max(N, M)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(max(1, i-w), min(i+w, M)+1):
            cost = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

    Dist = math.sqrt(D[N, M])

    return Dist


def ddtw(x, y, constraint=None, w= None):

    r"""One weakness of Dynamic Time Warping is the exclusive focus on values rather than "shape" 
    which results in misalignment. 
    Derivative Dynamic Time Warping (DDTW) [1]_ accounts for "shape" by replacing Euclidean in DTW 
    with squared difference of estimated derivatives of two points from different sequences. 
    The formula for esitmating derivatives are the following:

    .. math::

        
        \begin{equation}
            d[x_i] = \frac{(x_i-x_{i-1})+(x_{i+1}-x_{i-1})/2}{2} \\
            D(x_i, y_j) = (d[x_i]-d[y_j])^2
        \end{equation}
       

    Since the first and last elements in the sequence are undefined, 
    they are replaced by second and penultimate elements respectively. 


    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series. ``w`` defaults to ``None``, in which case the warping window is the length of the longest time series.
    :type w: float, optional
    :return: DDTW distance
    

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.elastic import ddtw
        import numpy as np

        X = np.array([3, 2, 4, 5, 5, 2, 4, 7, 9, 8])
        Y = np.array([3, 3, 3, 1, 6, 9, 9])

        ddtw_dist = ddtw(X, Y, 'Sakoe-Chiba', w = 3, fast = False)
        print(ddtw_dist)

    Output:

    .. code-block:: bash

      3.6571847095819483

    **References**

    .. [1] Eamonn J. Keogh and Michael J. Pazzani. “Derivative Dynamic Time
           Warping”. In: Proceedings of the 2001 SIAM International Conference on 
           Data Mining (SDM) (2001)    
    """

    if constraint == "None":
        return ddtw_n(x, y)
    elif constraint == "Sakoe-Chiba":
        return ddtw_scb(x, y, w)
        
def ddtw_n(x, y):
    y = np.squeeze(dev(np.array([y])))
    x = np.squeeze(dev(np.array([x])))
    N = len(x)
    M = len(y)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])
    Dist = math.sqrt(D[N, M])
    return Dist

def ddtw_scb(x, y, w):
    y = np.squeeze(dev(np.array([y])))
    x = np.squeeze(dev(np.array([x])))
    N = len(x)
    M = len(y)
    if w == None:
        w = max(N, M)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(max(1, i-w), min(i+w, M)+1):
            cost = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

    Dist = math.sqrt(D[N, M])
    return Dist

def lb_kim(y, x, fast = True):

    r"""LB_Kim [1]_ is the fastest DTW lower bound with :math:`O(1)` complexity, 
    defined as the following:

    .. math::

        \begin{equation}
            LB\_Kim(X, Y) = max\begin{cases}
            |X_1 - Y_1| \\
            |X_L - Y_L| \\
            |X_{max} - Y_{max}| \\
            |X_{min} - Y_{min}|
            \end{cases}
        \end{equation}
        
    
    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``
    :type fast: bool, optional
    :return: LB_Kim distance

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.elastic import lb_kim
        import numpy as np

        X = np.array([3, 2, 4, 5, 5, 2, 4, 7, 9, 8])
        Y = np.array([3, 3, 3, 1, 6, 9, 9])

        lb_kim_dist = lb_kim(X, Y)
        print(lb_kim_dist)

    Output:

    .. code-block:: bash

      1

    **References**

    .. [1] Sang-Wook Kim, Sanghyun Park, and Wesley W Chu. 2001. An index-based approach for similarity search supporting time warping in large sequence databases.
           In Data Engineering, 2001. Proceedings. 17th International Conference on. IEEE,607–614.

    """
    if fast == True:
        return lb_kim_numba(y, x)
    if fast == False:
        return lb_kim_n(y, x)

@jit(nopython = True)
def lb_kim_numba(y, x):
    lb_dist = max(abs(x[0] - y[0]),
                  abs(x[len(x)-1] - y[len(y)-1]),
                  abs(max(x)- max(y)),
                  abs(min(x)- min(y)))
    return lb_dist

def lb_kim_n(y, x):
    lb_dist = max(abs(x[0] - y[0]),
                  abs(x[len(x)-1] - y[len(y)-1]),
                  abs(max(x)- max(y)),
                  abs(min(x)- min(y)))
    return lb_dist


def lb_keogh (y, x, w = None, fast = True):

    r"""LB_Keogh [1]_ is one of the most popular dtw lower bounds. 
    LB_Keogh constructs upper and lower envelopes of query series and compute
    distance between these query envelopes and the candidate series, 
    while utilizing a window :math:``w`` to limit warping (shifted alignements).
    LB_Keogh is formally defined as:
 
    .. math::
        
        \begin{equation}
           LB\_Keogh(X, Y) = \sqrt{\sum_{i=1}^{L_X}\begin{cases}
            (Y_{i} - UE_{i} )^2 \mbox{ if } Y_{i} > UE_{i} \\
            (Y_{i} - LE_{i})^2 \mbox{ if } Y_{i} < LE_{i} \\
            0 \mbox{ otherwise  }
            \end{cases}}
        \end{equation}
        
    
    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param w: ``w`` is the largest temporal shift allowed between two time series, defaulting to ``None``
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_Keogh distance

    **Example:**
    
    Input:

    .. code-block:: python

        from elastic import lb_keogh
        import numpy as np

        X = np.array([3, 2, 14, 5, 35, 2, 4, 7, 19, 8])
        Y = np.array([3, -13, 3, 1, -6, 9, 9])

        lb_keogh_dist = lb_keogh(Y, X, 4)
        print(lb_keogh_dist)


    Output:

    .. code-block:: bash

      17.029386365926403

    **References**

    .. [1] Eamonn Keogh and Chotirat Ann Ratanamahatana. 2005. Exact indexing of
           dynamic time warping. Knowledge and Information Systems 7, 3 (2005), 358–386.
           [49] Eamonn J. Keogh and Michael J. Pazzani. 2001. Derivative Dynamic Time Warping.
           Proceedings of the 2001 SIAM International Conference on Data Mining (SDM)
           (2001).

    """

    if fast == True:
        return lb_keogh_numba(y, x, w)
    if fast == False:
        return lb_keogh_n(y, x, w)

@jit(nopython = True)
def lb_keogh_numba(y, x, w): 
    leny = len(y)
    lenx = len(x)
    if w == None:
        w = max(lenx, leny)
    lb_dist = 0
    for i in range(leny):
        wmin = max(0, i-w)
        wmax = min(i+w, lenx-1)
        UE = max(x[wmin : wmax +1])
        LE = min(x[wmin : wmax +1])
        if y[i] > UE:
            lb_dist += (y[i] - UE) ** 2
        if y[i] < LE:
            lb_dist += (y[i] - LE) ** 2
    return math.sqrt(lb_dist)

def lb_keogh_n(y, x, w): 
    leny = len(y)
    lenx = len(x)
    if w == None:
        w = max(lenx, leny)
    lb_dist = 0
    for i in range(leny):
        wmin = max(0, i-w)
        wmax = min(i+w, lenx-1)
        UE = max(x[wmin : wmax +1])
        LE = min(x[wmin : wmax +1])
        if y[i] > UE:
            lb_dist += (y[i] - UE) ** 2
        if y[i] < LE:
            lb_dist += (y[i] - LE) ** 2
    return math.sqrt(lb_dist)

def dev(X): # helper function of lb_ddtw
    lenx = X.shape[1]
    dX = (2 * X[:, 1:lenx-1] + X[:, 2:lenx] - 3*X[:, 0:lenx-2])/4
    first_col = np.array([dX[:, 0]])
    last_col = np.array([dX[:, dX.shape[1]-1]])
    dX = np.concatenate((first_col.T, dX), axis = 1)
    dX = np.concatenate((dX, last_col.T), axis =1)
    return dX


def lb_new(y, x, w = None, fast = True):

    r"""LB_New [1]_ is a tighter and more expensive lower bound of DTW relative to LB_Keogh.
    LB_New is formally defined as:

    .. math::

        \begin{equation}
            LB\_New = \sqrt{(x_1 - y_1)^2 + (x_{L_X}-y_{L_Y})^2 + \sum_{j=2}^{L_x-1}\delta(x_j, Y_i)}
        \end{equation}

    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param w: ``w`` is the largest temporal shift allowed between two time series, defaulting to ``None``
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_New distance

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.elastic import lb_new
        import numpy as np

        X = np.array([3, 2, 14, 5, 35, 2, 4, 7, 19, 8])
        Y = np.array([3, -13, 3, 1, -6, 9, 9])

        lb_new_dist = lb_new(Y, X, 4)
        print(lb_new_dist)

    Output:

    .. code-block:: bash

      17.08800749063506



    **References**

    .. [1] Yilin Shen, Yanping Chen, Eamonn Keogh, and Hongxia Jin. 2018. Accelerating
           time series searching with large uniform scaling. In Proceedings of the 2018 SIAM
           International Conference on Data Mining. SIAM, 234–242.
    """
    if fast == True:
        return lb_new_numba(y, x, w)
    if fast == False:
        return lb_new_n(y, x, w)


def lb_new_n(y, x, w = None):
    
    leny = len(y)
    lenx = len(x)
    if w == None:
        w = max(lenx, leny)
    lb_dist = (x[0]-y[0]) ** 2 + (x[lenx-1] - y[leny-1]) **2
    for i in range(1,leny-1):
        wmin = max(0, i - w)
        wmax = min(lenx - 1, i + w) 
        wx = np.array([i for i in x[wmin : wmax + 1]])
        Y = np.full(wx.shape[0], -y[i])
        diff = np.add(wx, Y)
        cost = min(np.square(diff))
        lb_dist = lb_dist + cost

    return math.sqrt(lb_dist)

@jit(nopython = True)
def lb_new_numba(y, x, w = None):
    
    leny = len(y)
    lenx = len(x)
    if w == None:
        w = max(lenx, leny)
    lb_dist = (x[0]-y[0]) ** 2 + (x[lenx-1] - y[leny-1]) **2
    for i in range(1,leny-1):
        wmin = max(0, i - w)
        wmax = min(lenx - 1, i + w) 
        wx = np.array([i for i in x[wmin : wmax + 1]])
        Y = np.full(wx.shape[0], -y[i])
        diff = np.add(wx, Y)
        cost = min(np.square(diff))
        lb_dist = lb_dist + cost

    return math.sqrt(lb_dist)

def lb_improved(x, y, w = None, fast = True):

    r"""LB_Improved [1]_ obtains greater pruning power over LB_Keogh
    by performing LB_Keogh twice. LB_Improved computes ordinary LB_Keogh as well as LB_Keogh between query
    series and the projection of candidate series on the query series
    envelopes. 

    .. math::

        \begin{equation}
            H(X, Y)_i = \begin{cases}
            UE(Y)_i \textup{ if } x_i \geq UE(Y)_i \\
            LE(Y)_i \textup{ if } x_i \leq LE(Y)_i \\
            x_i \textup{ otherwise }
        \end{cases}
        \end{equation}

    .. math::
        \begin{equation}
            LB\_Improved(X, Y) = LB\_Keogh(X, Y) + LB\_Keogh(Y, H(X, Y))
        \end{equation}
    

    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param w: ``w`` is the largest temporal shift allowed between two time series, defaulting to ``None``
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_Improved distance

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.elastic import lb_improved
        import numpy as np

        X = np.array([3, 2, 14, 5, 35, 2, 4])
        Y = np.array([3, -13, 3, 1, -6, 9, 9])

        lb_improved_dist = lb_improved(Y, X, 4)
        print(lb_improved_dist)

    Output:

    .. code-block:: bash

      31.480152477394387



    **References**

    .. [1] Daniel Lemire. 2009. Faster retrieval with a two-pass dynamic-time-warping
           lower bound. Pattern recognition 42, 9 (2009), 2169–2180.

    """
    if fast == True:
        return lb_improved_numba(y, x, w)
    if fast == False:
        return lb_improved_n(y, x, w)

# 3 following functions are helper functions of lb_improved
@jit(nopython = True)
def lower_b(t,w):
  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = min(t[max(0,i-w):min(len(t)-1,i+w)+1]);
  return b

@jit(nopython = True)
def upper_b(t,w):
  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = max(t[max(0,i-w):min(len(t)-1,i+w)+1])
  return b
    
@jit(nopython = True)
def lb_keogh_square(x,u,l):
  sumd = 0
  for i in range(len(x)):
    if x[i] > u[i]:
      sumd += (x[i] - u[i]) ** 2
    if x[i] < l[i]:
      sumd += (x[i] - l[i]) ** 2
  return sumd 

@jit(nopython = True)
def lb_improved_numba(x,y,w = None):
    if w == None:
        w = max(len(x), len(y))
    YUE, YLE = make_envelopes(y, w)
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

    return math.sqrt(lb_keogh_square(x,u,l) + lb_keogh_square(y,upper_h,lower_h))

def lower_b_n(t,w):
  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = min(t[max(0,i-w):min(len(t)-1,i+w)+1]);
  return b

def upper_b_n(t,w):
  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = max(t[max(0,i-w):min(len(t)-1,i+w)+1])
  return b
    
def lb_keogh_square_n(x,u,l):
  sumd = 0
  for i in range(len(x)):
    if x[i] > u[i]:
      sumd += (x[i] - u[i]) ** 2
    if x[i] < l[i]:
      sumd += (x[i] - l[i]) ** 2
  return sumd 

def lb_improved_n(x,y,w = None):
    if w == None:
        w = max(len(x), len(y))
    YUE, YLE = make_envelopes(y, w)
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
    upper_h = upper_b_n(h,w)
    lower_h = lower_b_n(h,w)

    return math.sqrt(lb_keogh_square_n(x,u,l) + lb_keogh_square_n(y,upper_h,lower_h))


# End of DTW

def lcss(x,y,epsilon,w = 100, constraint=None, fast=True):

    r"""
    Longest Common Subsequence (LCSS) [1]_ defines similarity by counting the number of "matches" between two time series, 
    where a match is added when the difference between two elements is less than a constant matching threshold, 
    :math:`\epsilon`, which must be tuned to each dataset. 

    This approach allows time series to stretch and match without rearranging the sequence of elements. 
    Note that LCSS allows some elements to be unmatched, whereas Dynamic Time Warping (DTW) pairs all elements, even the outliers. 
    This property enables LCSS to focus only on similar subsequences (matched elements) 
    in computing similarity and thus makes LCSS more robust to extremely noisy data with many disimilar outliers.

    LCSS differs from the general formula as if it determines a match, 
    it will compute :math:`1 + d_{(i-1,j-1)}` without considering the other directions, 
    and if it determines a miss then it will compute :math:`1 + min(d_{(i-1,j)},d_{(i,j-1))}`. 
    The complete formula for LCSS is: 

    .. math::
 
        \begin{aligned}
            D^u(x,y,\epsilon) & =
            \begin{cases}
                1 & \text{if $|x - y| <= \epsilon$} \\
                0 & \text{else}
            \end{cases}\\
            D^h(x,y,\epsilon) = D^v(x,y,\epsilon) & =
            \begin{cases}
                \infty & \text{if $|x - y|<= \epsilon$} \\
                0 & \text{else}
            \end{cases}\\
            D^d(x,y,\epsilon) & =
            \begin{cases}
                1 & \text{if $|x - y| <= \epsilon$} \\
                \infty & \text{else}
            \end{cases}\\
            \pi(d_{n,m}) & = 1 - \frac{d_{n,m}}{min(n,m)}
        \end{aligned}
    

   The function :math:`\pi` is the percent of elements in the smaller time series that are not part of the longest common subsequence. 
   Thus, if the final output is zero, every element in the shorter time series was part of the longest common subsequence. 
   If the value is one, the longest common subsequence was zero and there is no match between two series. 
   Thus, the range of values of the LCSS is :math:`[0,1]`.
   
   :param X: a time series
   :type X: np.array
   :param Y: another time series
   :type Y: np.array
   :param epsilon: the matching threshold
   :type epsilon: float
   :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
   :type constraint: float, optional
   :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
   :type w: float, optional
   :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
   :type fast: bool, optional
   :return: LCSS distance

    **Example:**

    Input:

    .. code-block:: python

        from tsdistance.ElasticMeasures import lcss
        import numpy as np

        X = np.array([3, 4, 38, 4, 5])
        Y = np.array([0, 3, 4])

        lcss_dist = lcss(X, Y, epsilon = 0.7)
        lcss_dist

    Output:

    .. code-block:: bash

        0.33333333333333337
    
   
   **References**

   .. [1] Michail  Vlachos,  George  Kollios,  and  Dimitrios  Gunopulos.  
          “DiscoveringSimilar Multidimensional Trajectories”. 
          In:Proceedings of the 18th Inter-national Conference on Data Engineering. IEEE Computer Society, USA. (2002)
    """

    if constraint == "None":
        if fast == True:
            return lcss_n_numba(x, y, epsilon)
        if fast == False:
            return lcss_n(x, y, epsilon);
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return lcss_scb_numba(x, y, epsilon,w)
        if fast == False:
            return lcss_scb(x, y, epsilon, w)

def lcss_n(x, y, epsilon):
    lenx = len(x)
    leny = len(y)
    D = np.zeros((lenx, leny))
    for i in range(lenx):
        wmin = 0
        wmax = leny-1
        for j in range(wmin, wmax+1):
            if i + j == 0:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else:
                    D[i][j] = 0
            elif i == 0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i][j-1]
            elif j ==0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i-1][j]
            else:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = max(D[i-1][j-1]+1,
                                  D[i-1][j],
                                  D[i][j+1])
                else:
                    D[i][j] = max(D[i-1][j-1],
                                  D[i-1][j],
                                  D[i][j+1])
    result = D[lenx-1, leny -1]
    return 1 - result/min(len(x),len(y))

def lcss_scb(x, y, epsilon, w):
    lenx = len(x)
    leny = len(y)
    D = np.zeros((lenx, leny))
    for i in range(lenx):
        wmin = max(0, i-w)
        wmax = min(leny-1, i+w)
        for j in range(wmin, wmax+1):
            if i + j == 0:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else:
                    D[i][j] = 0
            elif i == 0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i][j-1]
            elif j ==0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i-1][j]
            else:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = max(D[i-1][j-1]+1,
                                  D[i-1][j],
                                  D[i][j+1])
                else:
                    D[i][j] = max(D[i-1][j-1],
                                  D[i-1][j],
                                  D[i][j+1])
    result = D[lenx-1, leny -1]
    return 1 - result/min(len(x),len(y))

@jit(nopython=True)
def lcss_n_numba(x, y, epsilon):
    lenx = len(x)
    leny = len(y)
    D = np.zeros((lenx, leny))
    for i in range(lenx):
        wmin = 0
        wmax = leny-1
        for j in range(wmin, wmax+1):
            if i + j == 0:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else:
                    D[i][j] = 0
            elif i == 0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i][j-1]
            elif j ==0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i-1][j]
            else:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = max(D[i-1][j-1]+1,
                                  D[i-1][j],
                                  D[i][j+1])
                else:
                    D[i][j] = max(D[i-1][j-1],
                                  D[i-1][j],
                                  D[i][j+1])
    result = D[lenx-1, leny -1]
    return 1 - result/min(len(x),len(y))

@jit(nopython=True)
def lcss_scb_numba(x, y, epsilon, w):
    lenx = len(x)
    leny = len(y)
    D = np.zeros((lenx, leny))
    for i in range(lenx):
        wmin = max(0, i-w)
        wmax = min(leny-1, i+w)
        for j in range(wmin, wmax+1):
            if i + j == 0:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else:
                    D[i][j] = 0
            elif i == 0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i][j-1]
            elif j ==0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i-1][j]
            else:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = max(D[i-1][j-1]+1,
                                  D[i-1][j],
                                  D[i][j+1])
                else:
                    D[i][j] = max(D[i-1][j-1],
                                  D[i-1][j],
                                  D[i][j+1])
    result = D[lenx-1, leny -1]
    return 1 - result/min(len(x),len(y))
            


def dlcss(x, y, epsilon, constraint=None, w= None):

    r"""Similar to Derivative Dynamic Time Warping (DDTW), 
    Derivative Longest Common Subsequence (DLCSS) [1]_ enables the comparison of "general shape" 
    of two sequences rather than strict value comparisons. 
    Different from DDTW which replaces Euclidean Distance with estimated derivatives, 
    DLCSS considers both values of time series and values of derivatives. 
    The weights of these two components are parameterized. 
    [1]_ proposed two versions of DLCSS: 2D method that includes values and first derivatives, and 3D method that includes values, first derivatives, and the second derivatives. 
    The 2D method is defined as follows:

    .. math::

        
        \begin{equation*}
            DD_{LCSS}(x,y) = aLCSS(x, y) + bLCSS(\nabla{x},\nabla{y}) \\
            \nabla{x_i} = x(i+1) - x(i)
        \end{equation*}
            
    .. math::

        \textup{ for } i = 1, 2, ..., n-1\\
        a = 1-\alpha, b = \alpha, \textup{ for } \alpha \in [0,1]
        
    where :math:`\nabla{x}` and :math:`\nabla{y}` are the first discrete derivatives of :math:`(x,y)`, and :math:`a,b\in[0,1]` are parameters. 
    
    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param epsilon: the matching threshold
    :type epsilon: float
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series. ``w`` defaults to ``None``, in which case the warping window is the length of the longest time series.
    :type w: float, optional
    :return: DLCSS distance
    

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.elastic import ddtw
        import numpy as np

        X = np.array([3, 2, 4, 5, 5, 2, 4, 7, 9, 8])
        Y = np.array([3, 3, 3, 1, 6, 9, 9])

        ddtw_dist = ddtw(X, Y, 'Sakoe-Chiba', w = 3, fast = False)
        print(ddtw_dist)

    Output:

    .. code-block:: bash

      3.6571847095819483

    **References**

    .. [1] Tomasz Górecki. “Using derivatives in a longest common subsequence dis-
           similarity measure for time series classification”. 
           In: Pattern Recognition Letters (2014).
  
    """

    if constraint == "None":
        return dlcss_n(x, y, epsilon)
    elif constraint == "Sakoe-Chiba":
        return ddtw_scb(x, y, epsilon, w)

def dlcss_n(x, y, epsilon):
    lenx = len(x)
    leny = len(y)
    y = np.squeeze(dev(np.array([y])))
    x = np.squeeze(dev(np.array([x])))
    D = np.zeros((lenx, leny))
    for i in range(lenx):
        wmin = 0
        wmax = leny-1
        for j in range(wmin, wmax+1):
            if i + j == 0:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else:
                    D[i][j] = 0
            elif i == 0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i][j-1]
            elif j ==0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i-1][j]
            else:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = max(D[i-1][j-1]+1,
                                  D[i-1][j],
                                  D[i][j+1])
                else:
                    D[i][j] = max(D[i-1][j-1],
                                  D[i-1][j],
                                  D[i][j+1])
    result = D[lenx-1, leny -1]
    return 1 - result/min(len(x),len(y))

def dlcss_scb(x, y, epsilon, w):
    y = np.squeeze(dev(np.array([y])))
    x = np.squeeze(dev(np.array([x])))
    lenx = len(x)
    leny = len(y)
    if w == None:
        w = max(lenx, leny)
    D = np.zeros((lenx, leny))
    for i in range(lenx):
        wmin = max(0, i-w)
        wmax = min(leny-1, i+w)
        for j in range(wmin, wmax+1):
            if i + j == 0:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else:
                    D[i][j] = 0
            elif i == 0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i][j-1]
            elif j ==0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i-1][j]
            else:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = max(D[i-1][j-1]+1,
                                  D[i-1][j],
                                  D[i][j+1])
                else:
                    D[i][j] = max(D[i-1][j-1],
                                  D[i-1][j],
                                  D[i][j+1])
    result = D[lenx-1, leny -1]
    return 1 - result/min(len(x),len(y))

def lb_keogh_lcss(y, x, epsilon, w = None, fast = True):
    r"""The LCSS lower bound, LB_Keogh_LCSS [1]_ is adapted from
    LB_Keogh [2]_ and considers the LCSS matching threshold
    :math:`epsilon`in constructing envelopes.

    .. math::

        \begin{equation}
            LB\_LCSS(X, Y) = 1 - \frac{1}{L_Y}\overset{L_Y}{\underset{i=1}{\sum}}\begin{cases}
            1 \mbox{ if }  LE_i \leq{Y_i} \leq{UE_i} \\
            0 \mbox{ otherwise }
            \end{cases}
        \end{equation}

    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param epsilon: the matching threshold
    :type epsilon: float
    :param w: ``w`` is the largest temporal shift allowed between two time series, defaulting to ``None``
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_Keogh_LCSS distance

    **Example:**
    
    Input:

    .. code-block:: python

        from elastic import lb_keogh_lcss
        import numpy as np

        X = np.array([3, 2, 14, 5, 35, 2, 4])
        Y = np.array([3, -13, 3, 1, -6, 9, 9])
        lb_keogh_lcss_dist = lb_keogh_lcss(Y, X, 4)
        print(lb_keogh_lcss_dist)

    Output:

    .. code-block:: bash

      0.2857142857142857


    **References**

    .. [1] Chang Wei Tan, François Petitjean, and Geoffrey I Webb. 2020. FastEE: Fast
           Ensembles of Elastic Distances for time series classification. Data Mining and
           Knowledge Discovery 34, 1 (2020), 231–272.
    
    .. [2] Eamonn Keogh and Chotirat Ann Ratanamahatana. 2005. Exact indexing of
           dynamic time warping. Knowledge and Information Systems 7, 3 (2005), 358–386.
    """
    if fast == True:
        return lb_keogh_lcss_numba(y, x, epsilon, w)
    if fast == False:
        return lb_keogh_lcss_n(y, x, epsilon, w)

def lb_keogh_lcss_n(y, x, epsilon, w):
    leny = len(y)
    if w == None:
        w = leny
    XUE, XLE = make_envelopes(x, w)
    LE_lower = np.subtract(XLE,epsilon)
    UE_higher = np.add(XUE,epsilon)
    sum = 0
    for i in range(leny):
        if y[i] >= LE_lower[i] and y[i] <= UE_higher[i]:
            sum += 1
    lb_dist = 1 - (sum/(min(len(x),len(y))))
    return lb_dist

@jit(nopython = True)
def lb_keogh_lcss_numba(y, x, epsilon, w):
    leny = len(y)
    if w == None:
        w = leny
    XUE, XLE = make_envelopes(x, w)
    LE_lower = np.subtract(XLE,epsilon)
    UE_higher = np.add(XUE,epsilon)
    sum = 0
    for i in range(leny):
        if y[i] >= LE_lower[i] and y[i] <= UE_higher[i]:
            sum += 1
    lb_dist = 1 - (sum/(min(len(x),len(y))))
    return lb_dist

# End of LCSS

# Start of ERP

def erp(x, y, m, constraint=None, w= None, fast = True):

    r"""
    Edit Distance with Real Penalty (ERP) [1]_ is another edit distance measure that aims to take the advantages of eing a metric (like most $L_{p}$ norm measures) and allowing temporal shifts. 
    It does this by using Lp-norm distance metrics when comparing two elements or comparing each element to a gap variable, m. 
    Being a metric, ERP makes lower bounding possible through the triangle inequality. 
    This is very useful for pruning through clustering and classfication algorithms.

    ERP provides an advantage over other edit distance measures by providing exact differences between values. 
    Additionally, ERP has no :math:`\epsilon` value to tune. 
    Instead, one has to set a gap variable which is often set to 0 to provide intuitive results.

    Lastly, ERP is also very editable; 
    in the formula below, Euclidean distance is used as the internal distance measure but other measures such as absolute difference are also compatible with ERP. 
    This would change :math:`D` and :math:`\pi` but other metrics might have more desirable properties to certain users.

    .. math::

        \begin{aligned}
            D^u & = 0\\
            D^h(x,y,m) & = (x - m)^2\\
            D^v(x,y,m) & = (y - m)^2\\
            D^d(x,y,m) & = (x - y)^2\\
            \pi(d_{n,m}) & = \sqrt{d_{n,m}}
        \end{aligned}

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param m: the gap variable
    :type m: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; ``w`` defaults to ``None``, in which case the warping window is the length of the longest time series.
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: ERP distance

    **Example:**

    Input:

    .. code-block:: python

            from tsdistance.ElasticMeasures import lcss
            import numpy as np

            X = np.array([3, 4, 38, 4, 5])
            Y = np.array([0, 3, 4])

            erp_dist = erp(X, Y, m = 0)
            erp_dist

    Output:

    .. code-block:: bash

        34.61213659975356

    **References**

    .. [1] Lei  Chen  and  Raymond  Ng.  “On  The  Marriage  of  Lp-norms  and  EditDistance”. In:Proceedings of the 30th VLDB Conference,Toronto, Canada. (2004)
    
    
    """

    if constraint == "None":
        if fast == True:
            return erp_n_numba(x, y, m)
        if fast == False:
            return erp_n(x, y, m)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return erp_scb_numba(x, y, m, w)
        if fast == False:
            return erp_scb(x, y, m, w)


@jit(nopython=True)
def erp_n_numba(x, y, m):
    lenx = len(x)
    leny = len(y)

    acc_cost_mat = np.full((lenx, leny), np.inf)

    for i in range(lenx):
        minw = 0
        maxw = leny-1
        
        for j in range(minw, maxw+1):
            if i + j == 0:
                acc_cost_mat[i, j] = 0
            elif i == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + (y[j]-m)**2
            elif j == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i-1, j] + (x[i]-m)**2
            else:
                acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + (x[i] - y[j])**2,
                                         acc_cost_mat[i, j-1] + (y[j] - m)**2,
                                         acc_cost_mat[i-1, j] + (x[i]-m)**2)
    
    return math.sqrt(acc_cost_mat[lenx-1, leny-1])

def erp_n(x, y, m):
    lenx = len(x)
    leny = len(y)

    acc_cost_mat = np.full((lenx, leny), np.inf)

    for i in range(lenx):
        minw = 0
        maxw = leny-1
        
        for j in range(minw, maxw+1):
            if i + j == 0:
                acc_cost_mat[i, j] = 0
            elif i == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + (y[j]-m)**2
            elif j == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i-1, j] + (x[i]-m)**2
            else:
                acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + (x[i] - y[j])**2,
                                         acc_cost_mat[i, j-1] + (y[j] - m)**2,
                                         acc_cost_mat[i-1, j] + (x[i]-m)**2)
    
    return math.sqrt(acc_cost_mat[lenx-1, leny-1])


def erp_scb(x, y, m, w):
    lenx = len(x)
    leny = len(y)

    if w == None:
        w = max(lenx, leny)

    acc_cost_mat = np.full((lenx, leny), np.inf)

    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(leny-1, i + w)
        
        for j in range(minw, maxw+1):
            if i + j == 0:
                acc_cost_mat[i, j] = 0
            elif i == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + (y[j]-m)**2
            elif j == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i-1, j] + (x[i]-m)**2
            else:
                acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + (x[i] - y[j])**2,
                                         acc_cost_mat[i, j-1] + (y[j] - m)**2,
                                         acc_cost_mat[i-1, j] + (x[i]-m)**2)
    
    return math.sqrt(acc_cost_mat[lenx-1, leny-1])

@jit(nopython=True)
def erp_scb_numba(x, y, m, w):
    lenx = len(x)
    leny = len(y)
    
    if w == None:
        w = max(lenx, leny)

    acc_cost_mat = np.full((lenx, leny), np.inf)

    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(leny-1, i + w)
        
        for j in range(minw, maxw+1):
            if i + j == 0:
                acc_cost_mat[i, j] = 0
            elif i == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + (y[j]-m)**2
            elif j == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i-1, j] + (x[i]-m)**2
            else:
                acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + (x[i] - y[j])**2,
                                         acc_cost_mat[i, j-1] + (y[j] - m)**2,
                                         acc_cost_mat[i-1, j] + (x[i]-m)**2)
    
    return math.sqrt(acc_cost_mat[lenx-1, leny-1])



def lb_erp(x, y, fast = True):
    r"""LB_ERP [1]_ is a ERP lower bound that utilizes the triangular inequality, 
    formally defined as:

    .. math::

        \begin{equation}
            LB\_ERP(X, Y) = |\sum{Y} - \sum{X}|
        \end{equation}

    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_ERP distance

    **Example:**
    
    Input:

    .. code-block:: python

        from elastic import lb_erp
        import numpy as np

        X = np.array([3, 2, 14, 5, 35, 2, 4])
        Y = np.array([3, -13, 3, 1, -6, 9, 9])
        lb_erp_dist = lb_erp(Y, X)
        print(lb_erp_dist)

    Output:

    .. code-block:: bash

      59


    **References**

    .. [1] Lei Chen and Raymond Ng. 2004. On The Marriage of Lp-norms and Edit Distance.
           Proceedings of the 30th VLDB Conference,Toronto, Canada (2004).

    """

    if fast == True:
        return lb_erp_numba(x, y)
    if fast == False:
        return lb_erp_n(x, y)

def lb_erp_n(x, y): # LB_ERP
    return abs(np.sum(x) - np.sum(y))

@jit(nopython = True)
def lb_erp_numba(x, y): # LB_ERP
    return abs(np.sum(x) - np.sum(y))


def lb_keogh_erp(y, x, m, w = None, fast = True):

    r"""LB_Keogh_ERP [1]_ is a ERP lower bound adapted from LB_Keogh for DTW.
    LB_Keogh_ERP is formally defined as:

    .. math::
        
        UE_i = max(g, max(c_{i-w}:c_{i+w})) 
            
    .. math::
        LE_i = min(g, min(c_{i-w}:c_{i+w}))
    
    .. math::
        LB\_Keogh_ERP(X, Y) =  \sqrt{\sum_{i=1}\begin{cases}
                                                    (Y_{i} - UE_{i} )^2 \textup{ if } Y_{i} > UE_{i} \\
                                                    (Y_{i} - LE_{i})^2 \textup{ if } Y_{i} < LE_{i} \\
                                                    0 \textup{ otherwise }
                                                    \end{cases}}
        
    
    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param m: the gap variable
    :type m: float
    :param w: ``w`` is the largest temporal shift allowed between two time series, defaulting to ``None``
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_Keogh_ERP distance

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.elastic import lb_keogh_erp
        import numpy as np

        X = np.array([3, 2, 14, 5, 35, 2, 4])
        Y = np.array([3, -13, 3, 1, -6, 9, 9])
        lb_keogh_erp_dist = lb_keogh_erp(Y, X, m = 0)
        print(lb_keogh_erp_dist)

    Output:

    .. code-block:: bash

      14.317821063276353


    **References**

    .. [1] Chang Wei Tan, François Petitjean, and Geoffrey I Webb. 2020. FastEE: Fast
           Ensembles of Elastic Distances for time series classification. Data Mining and
           Knowledge Discovery 34, 1 (2020), 231–272.

    """

    if fast == True:
        return lb_keogh_erp_numba(y, x, m, w)
    if fast == False:
        return lb_keogh_erp_n(y, x, m, w)

def lb_keogh_erp_n(y, x, m, w):
    leny = len(y)
    lenx = len(x)

    if w == None:
        w = max(lenx, leny)
    
    XUE, XLE = make_envelopes(x, w)
    

    lb_dist = 0
    for i in range(leny):
        UE = max(m, XUE[i])
        LE = min(m, XLE[i])

        if y[i] > UE:
            lb_dist += (y[i] - UE) ** 2
        if y[i] < LE:
            lb_dist += (y[i] - LE) ** 2

    return math.sqrt(lb_dist)

@jit(nopython = True)
def lb_keogh_erp_numba(y, x, m, w):
    leny = len(y)
    lenx = len(x)

    if w == None:
        w = max(lenx, leny)
    
    XUE, XLE = make_envelopes(x, w)
    
    lb_dist = 0
    for i in range(leny):
        UE = max(m, XUE[i])
        LE = min(m, XLE[i])

        if y[i] > UE:
            lb_dist += (y[i] - UE) ** 2
        if y[i] < LE:
            lb_dist += (y[i] - LE) ** 2

    return math.sqrt(lb_dist)


def lb_kim_erp(y, x, m, fast = True):

    r"""LB_Kim_ERP [1]_ is a ERP lower bound adapted from LB_Kim orginally developed for DTW.
    LB_Kim_ERP is formally defined as:

    .. math::

        
        \begin{equation}
            LB\_Kim-ERP(X, Y) = max\begin{cases}
                                            |X'_1 - Y'_1| \\
                                            |X'_L - Y'_L| \\
                                            |X'_{max} - Y'_{max}| \\
                                            |X'_{min} - Y'_{min}|
                                            \end{cases}
        \end{equation}
       

    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param m: the gap variable
    :type m: float
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_Kim_ERP distance

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.elastic import lb_kim_erp
        import numpy as np

        X = np.array([3, 2, 14, 5, 35, 2, 4])
        Y = np.array([3, -13, 3, 1, -6, 9, 9])
        lb_kim_erp_dist = lb_kim_erp(Y, X, m = 0)
        print(lb_kim_erp_dist)

    Output:

    .. code-block:: bash

      26


    **References**

    .. [1] Chang Wei Tan, François Petitjean, and Geoffrey I Webb. 2020. FastEE: Fast
           Ensembles of Elastic Distances for time series classification. Data Mining and
           Knowledge Discovery 34, 1 (2020), 231–272.
    
    """
    if fast == True:
        return lb_kim_erp_numba(y, x, m)
    if fast == False:
        return lb_kim_erp_n(y, x, m)

def lb_kim_erp_n(y, x, m): 
    x_max = max(m, max(x))
    x_min = min(m, min(x))
    lb_dist = max(abs(x[0]-y[0]),
                  abs(x[len(x)-1] - y[len(y)-1]),
                  abs(x_max - max(y)),
                  abs(x_min - min(y)))

    return lb_dist

@jit(nopython = True)
def lb_kim_erp_numba(y, x, m): 
    x_max = max(m, max(x))
    x_min = min(m, min(x))
    lb_dist = max(abs(x[0]-y[0]),
                  abs(x[len(x)-1] - y[len(y)-1]),
                  abs(x_max - max(y)),
                  abs(x_min - min(y)))

    return lb_dist


# End of ERP

# Start of EDR
def edr(x, y, m=0, constraint=None, w=100, fast = True):

    r"""
    Edit Distance on Real Sequences (EDR) [1]_ is an edit-based elastic measure. 
    Compared to Longest Common Subsequence (LCSS), EDR does not discriminate which direction to pick based on if the current elements were considered a match. 
    Therefore, it is possible for the current elements to match and for the algorithm to take a horizontal or vertical step which is not possible in LCSS. 
    The intuition behind this method is that EDR aims to capture how many edit operations (delete, insert, substitute) it takes to change one time series into another. 
    To determine if one element is the same as another, a matching threshold :math:`\epsilon` is used in a similar way to LCSS, where a match is added when the difference between two comparing elements is less than :math:`\epsilon`.
    The threshold's ability to quantize differences between comparing elements makes EDR useful for very noisy data as outliers in dataset won't disrupt the overall pattern.

    .. math::
        \begin{aligned}
            D(x,y,\epsilon) & =
            \begin{cases}
                0 & \text{if $|x - y| <= \epsilon$}\\
                1 & \text{else}
            \end{cases}\\
            \pi(d_{n,m}) & = d_{n,m}
        \end{aligned}

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param m: the matching threshold, default to 0
    :type m: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100.
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: EDR distance

    **Example:**

    Input:

    .. code-block:: python

        from tsdistance.ElasticMeasures import edr 
        import numpy as np

        X = np.array([3, 4, 74, 4, 5])
        Y = np.array([0, 3, 4])

        edr_dist = erp(X, Y, m = 4)
        edr_dist

    Output:

    .. code-block:: bash

        3.0

    **Reference**

    .. [1]  Lei Chen, M. Tamer Ozsu, and Vincent Oria. “Robust and Fast SimilaritySearch  for  Moving  Object  Trajectories”.  In:ACM SIGMOD, Baltimore,Maryland, USA(2005)
    
    """

    if constraint == "None":
        if fast == True:
            return edr_n_numba(x, y, m)
        if fast == False:
            return edr_n(x, y, m)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return edr_scb_numba(x, y, m, w)
        if fast == False:
            return edr_scb(x, y, m, w)


def edr_n(x, y, m):
    cur = np.full((1, len(y)), -np.inf)
    prev = np.full((1, len(y)), -np.inf)

    for i in range(len(x)):
        minw = 0
        maxw = len(y)-1
        prev = cur
        cur = np.full((1, len(y)), -np.inf)

        for j in range(int(minw), int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[i] - y[j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y) - 1]


@jit(nopython=True)
def edr_n_numba(x, y, m):
    cur = np.full((1, len(y)), -np.inf)
    prev = np.full((1, len(y)), -np.inf)

    for i in range(len(x)):
        minw = 0
        maxw = len(y)-1
        prev = cur
        cur = np.full((1, len(y)), -np.inf)

        for j in range(int(minw), int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[i] - y[j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y) - 1]

def edr_scb(x, y, m, w):
    lenx = len(x)
    leny = len(y)

    if w == None:
        w = max(lenx, leny)

    cur = np.full((1, leny), -np.inf)
    prev = np.full((1, leny), -np.inf)

    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(len(y)-1, i + w)
        prev = cur
        cur = np.full((1, leny), -np.inf)

        for j in range(int(minw), int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[i] - y[j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y) - 1]


@jit(nopython=True)
def edr_scb_numba(x, y, m, w):
    lenx = len(x)
    leny = len(y)

    if w == None:
        w = max(lenx, leny)

    cur = np.full((1, leny), -np.inf)
    prev = np.full((1, leny), -np.inf)

    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(len(y)-1, i + w)
        prev = cur
        cur = np.full((1, leny), -np.inf)

        for j in range(int(minw), int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[i] - y[j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y) - 1]

# End of EDR

# Start of TWED

@jit(nopython=True)
def dist(x, y):
    return (x - y) ** 2


def twed(x, timesx, y, timesy, lamb, nu, constraint=None, w= None, fast = True):

    r"""
    Time Warp Edit Distance (TWED) [1]_ is an elastic measure 
    that aims to combine the merits of DTW and edit distance measures like ERP. 
    Unlike ERP, 
    TWED uses time stamps as part of the algorithm which punishes elements that have very different time stamps. 
    TWED controls the extent of this punishment with the parameter :math:`\nu`. 
    TWED replaces the insert, delete, and replace with delete-X, delete-Y, and match.
    The delete operation has a cost of :math:`\lambda`. 
    TWED is a metric and gains all the benefits of a metric as long as the internal distance function is a metric such as absolute value or Euclidean. 
    However, TWED requires two parameters, :math:`\nu` and r:math:`\lambda`, to be set properly which depend on the distance measure. 
    Additionally, the use of time stamps might be difficult as not all data sets include time stamps for the data.

    .. math::

        \begin{aligned}
            D^u(x,\overline{x},t_x, \overline{t_x},y,\overline{y},t_y,\overline{t_y},\nu,\lambda) & = dist(x,y) \\
            D^v(x,\overline{x},t_x, \overline{t_x},y,\overline{y},t_y,\overline{t_y},\nu,\lambda) & = dist(y,\overline{y}) + \nu * (t_y - \overline{t_y}) + \lambda\\
            D^h(x,\overline{x},t_x, \overline{t_x},y,\overline{y},t_y,\overline{t_y},\nu,\lambda) & = dist(x,\overline{x}) + \nu * (t_x - \overline{t_x}) + \lambda\\
            D^d(x,\overline{x},t_x, \overline{t_x},y,\overline{y},t_y,\overline{t_y},\nu,\lambda) & = dist(x,y)+ dist(\overline{x}, \overline{y}) + \nu * (abs(t_y - \overline{t_y}) + abs(t_x - \overline{t_x}))\\
            \pi(d_{n,m}) & = d_{n,m}
        \end{aligned}

    :param X: a time series
    :type X: np.array
    :param timesx: time stamp of time series :math:`X`
    :type timesx: np.array
    :param Y: another time series 
    :type Y: np.array
    :param timesy: time stamp of time series :math:`Y`
    :type timesy: np.array
    :param lamb: cost of delete operation, :math:`\lambda`.
    :type lamb: float
    :param nu: cost of difference in timestamps, :math:`\nu`.
    :type nu: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; ``w`` defaults to ``None``, in which case the warping window is the length of the longest time series.
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: TWED distance

    **Example:**

    Input:

    .. code-block:: python

        X = np.array([3, 4, 76, 4, 5])
        Y = np.array([0, 3, 4])
        timesx = np.array([i for i in range(len(X))])
        timesy = np.array([i for i in range(len(Y))])

        twed_distance = twed(X, timesx, Y, timesy, lamb = 2.5, nu = 1, w = 5)
        print(twed_distance)

    Output:

    .. code-block:: bash

        4.5

    **References**

    .. [1] Pierre-Fran ̧cois  Marteau.  “Time  Warp  Edit  Distance  with  Stiffness  Ad-justment  for  Time  Series  Matching”.  In:IEEE Transactions on PatternAnalysis and Machine Intelligence31.306 - 318 (2009)
    """

    if constraint == "None":
        if fast == True:
            return twed_n_numba(x, timesx, y, timesy, lamb, nu)
        if fast == False:
            return twed_n(x, timesx, y, timesy, lamb, nu)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return twed_scb_numba(x, timesx, y, timesy, lamb, nu, w)
        if fast == False:
            return twed_scb(x, timesx, y, timesy, lamb, nu, w)


@jit(nopython=True)
def twed_n_numba(x,timesx, y, timesy, lamb, nu):
    xlen = len(x)
    ylen = len(y)
    cur = np.full(ylen, np.inf)
    prev = np.full(ylen, np.inf)
    for i in range(0, xlen):
        prev = cur
        cur = np.full(ylen, np.inf)
        minw = 0
        maxw = ylen-1
        for j in range(minw, maxw+1):
            if i + j == 0:
                cur[j] = (x[i] - y[j]) **2
            elif i == 0:
                c1 = (cur[j - 1]+ (y[j - 1] - y[j]) **2 + nu * (timesy[j] - timesy[j - 1])+ lamb)
                cur[j] = c1
            elif j == 0:
                c1 = (prev[j]+ (x[i - 1] - x[i]) **2+ nu * (timesx[i] - timesx[i - 1])+ lamb)
                cur[j] = c1
            else:
                c1 = (prev[j]+(x[i - 1] - x[i]) **2+ nu * (timesx[i] - timesx[i - 1])+ lamb)
                c2 = (cur[j - 1]+ (y[j - 1] - y[j])**2+ nu * (timesy[j] - timesy[j - 1])+ lamb)
                c3 = (prev[j - 1]+ (x[i] - y[j]) ** 2+ (x[i - 1]- y[j - 1]) ** 2+ nu* (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1])))
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]

def twed_n(x,timesx, y, timesy, lamb, nu):
    xlen = len(x)
    ylen = len(y)
    cur = np.full(ylen, np.inf)
    prev = np.full(ylen, np.inf)
    for i in range(0, xlen):
        prev = cur
        cur = np.full(ylen, np.inf)
        minw = 0
        maxw = ylen-1
        for j in range(minw, maxw+1):
            if i + j == 0:
                cur[j] = (x[i] - y[j]) **2
            elif i == 0:
                c1 = (cur[j - 1]+ (y[j - 1] - y[j]) **2 + nu * (timesy[j] - timesy[j - 1])+ lamb)
                cur[j] = c1
            elif j == 0:
                c1 = (prev[j]+ (x[i - 1] - x[i]) **2+ nu * (timesx[i] - timesx[i - 1])+ lamb)
                cur[j] = c1
            else:
                c1 = (prev[j]+(x[i - 1] - x[i]) **2+ nu * (timesx[i] - timesx[i - 1])+ lamb)
                c2 = (cur[j - 1]+ (y[j - 1] - y[j])**2+ nu * (timesy[j] - timesy[j - 1])+ lamb)
                c3 = (prev[j - 1]+ (x[i] - y[j]) ** 2+ (x[i - 1]- y[j - 1]) ** 2+ nu* (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1])))
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]
    

@jit(nopython=True)
def twed_scb_numba(x,timesx, y, timesy, lamb, nu, w=None):
    xlen = len(x)
    ylen = len(y)
    if w == None:
        w = max(xlen, ylen)
    cur = np.full(ylen, np.inf)
    prev = np.full(ylen, np.inf)
    for i in range(0, xlen):
        prev = cur
        cur = np.full(ylen, np.inf)
        minw = max(0, i - w)
        maxw = min(ylen-1, i + w)
        for j in range(minw, maxw+1):
            if i + j == 0:
                cur[j] = (x[i] - y[j]) **2
            elif i == 0:
                c1 = (cur[j - 1]+ (y[j - 1] - y[j]) **2 + nu * (timesy[j] - timesy[j - 1])+ lamb)
                cur[j] = c1
            elif j == 0:
                c1 = (prev[j]+ (x[i - 1] - x[i]) **2+ nu * (timesx[i] - timesx[i - 1])+ lamb)
                cur[j] = c1
            else:
                c1 = (prev[j]+(x[i - 1] - x[i]) **2+ nu * (timesx[i] - timesx[i - 1])+ lamb)
                c2 = (cur[j - 1]+ (y[j - 1] - y[j])**2+ nu * (timesy[j] - timesy[j - 1])+ lamb)
                c3 = (prev[j - 1]+ (x[i] - y[j]) ** 2+ (x[i - 1]- y[j - 1]) ** 2+ nu* (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1])))
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]


def twed_scb(x,timesx, y, timesy, lamb, nu, w=None):
    xlen = len(x)
    ylen = len(y)
    if w == None:
        w = max(xlen, ylen)
    cur = np.full(ylen, np.inf)
    prev = np.full(ylen, np.inf)
    for i in range(0, xlen):
        prev = cur
        cur = np.full(ylen, np.inf)
        minw = max(0, i - w)
        maxw = min(ylen-1, i + w)
        for j in range(minw, maxw+1):
            if i + j == 0:
                cur[j] = (x[i] - y[j]) **2
            elif i == 0:
                c1 = (cur[j - 1]+ (y[j - 1] - y[j]) **2 + nu * (timesy[j] - timesy[j - 1])+ lamb)
                cur[j] = c1
            elif j == 0:
                c1 = (prev[j]+ (x[i - 1] - x[i]) **2+ nu * (timesx[i] - timesx[i - 1])+ lamb)
                cur[j] = c1
            else:
                c1 = (prev[j]+(x[i - 1] - x[i]) **2+ nu * (timesx[i] - timesx[i - 1])+ lamb)
                c2 = (cur[j - 1]+ (y[j - 1] - y[j])**2+ nu * (timesy[j] - timesy[j - 1])+ lamb)
                c3 = (prev[j - 1]+ (x[i] - y[j]) ** 2+ (x[i - 1]- y[j - 1]) ** 2+ nu* (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1])))
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]


def lb_twed(y, x, lamb, nu, w = None, fast = True):
    r"""LB_TWED is an TWED lower bound which constructs upper and lower envelopes, 
    formally defined as:

    .. math::

       
        \begin{equation}
            FastEE\_TWED(X, Y) =  min\begin{cases}
                                (X_1-Y_1)^2 \\
                                X_1^2 + v + \lambda \\
                                Y_1^2 + v + \lambda  
                                \end{cases}  + \overset{L_Y}{\underset{i=2}{\sum}}\begin{cases}
                                min(v, (Y_i - max(X_{max}, Y_{i-1}))^2 \\ \mbox{ if } Y_i>max(X_{max}, Y_{i-1}) \\
                                min(v, (Y_i - min(X_{min}, Y_{i-1}))^2 \\ \mbox{ if } Y_i<min(X_{min}, Y_{i-1}) \\
                                0 \mbox{ otherwise }
                                \end{cases}
        \end{equation}
        

    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param lamb: cost of delete operation, :math:`\lambda`.
    :type lamb: float
    :param nu: cost of difference in timestamps, :math:`\nu`.
    :type nu: float
    :param w: ``w`` is the largest temporal shift allowed between two time series, defaulting to ``None``
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_TWED distance

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.elastic import lb_twed
        import numpy as np

        X = np.array([-3, 2, -4, -5, -3, -2, -93])
        Y = np.array([3, 13, 3, 1, 6, 9, 9])
        lb_twed_dist = lb_twed(Y, X, lamb = 1, nu = 0.01)
        print(lb_twed_dist)

    Output:

    .. code-block:: bash

      10.04


    **References**

    .. [1] Chang Wei Tan, François Petitjean, and Geoffrey I Webb. 2020. FastEE: Fast
           Ensembles of Elastic Distances for time series classification. Data Mining and
           Knowledge Discovery 34, 1 (2020), 231–272.
    
    
    """

    if fast == True:
        return lb_twed_numba(y, x, lamb, nu, w)
    if fast == False:
        return lb_twed_n(y, x, lamb, nu, w)

def lb_twed_n(y, x, lamb, nu, w):
    lenx = len(x)
    leny = len(y)

    if w == None:
        w = max(lenx, leny)

    XUE, XLE = make_envelopes(x, w)

    lb_dist = min((x[0] - y[0])**2, (x[0])**2 + nu + lamb, (y[0])**2 + nu + lamb)

    for i in range(1, leny):

        if y[i] > max(XUE[i], y[i-1]):
            lb_dist += min(nu, (y[i]- max(XUE[i], y[i-1]))**2)
        if y[i] < min(XLE[i], y[i-1]):
            lb_dist += min(nu, (y[i]- min(XLE[i], y[i-1]))**2)
        
    return lb_dist

@jit(nopython = True)
def lb_twed_numba(y, x, lamb, nu, w):
    lenx = len(x)
    leny = len(y)

    if w == None:
        w = max(lenx, leny)

    XUE, XLE = make_envelopes(x, w)

    lb_dist = min((x[0] - y[0])**2, (x[0])**2 + nu + lamb, (y[0])**2 + nu + lamb)

    for i in range(1, leny):

        if y[i] > max(XUE[i], y[i-1]):
            lb_dist += min(nu, (y[i]- max(XUE[i], y[i-1]))**2)
        if y[i] < min(XLE[i], y[i-1]):
            lb_dist += min(nu, (y[i]- min(XLE[i], y[i-1]))**2)
        
    return lb_dist

# End of TWED

# Start of MSM
def msm(x,y,c,constraint=None,w=None, fast = True):

    r"""Move-Split-Merge (MSM) [1]_ is an edit distance measure that deconstructs the popular editing operations (insert, delete, and substitute);
    instead it proposes sub-operations that have can be used in conjunctions to replicate the original operations. 
    Move functions identical to a substitute, changing the value of an element. 
    Merge combines two equal elements in a series into one. 
    Split takes an element creates a duplicate adjacent to it. 
    Thus, insert can be seen as a split-move operation and delete can be seen as a merge-move operation. 

    Similar to ERP, 
    MSM has the advantage of being a metric, 
    which allows MSM to be combined with other generic indexing, 
    clustering, and visualization methods designed to in any metric space. 
    However unlike ERP, MSM is invariant based on the choice of the origin. 
    This means that the distance calculated is unaffected by translations 
    (adding the same constant to both time series).

    Each operation has an associated cost:

    .. math::

        \begin{equation*}
            Cost(move) = |x - \overline{x}|
        \end{equation*}
    
    .. math::

        \begin{equation*}
            Cost(split) = Cost(merge) = c
        \end{equation*}
    
    where :math:`c` is a set constant, and :math:`\overline{x}` is the new value.

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param c: the cost for one *move* or *split** operation 
    :type c: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; ``w`` defaults to ``None``, in which case the warping window is the length of the longest time series.
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: MSM distance

    **Example:**

    Input:

    .. code-block:: python

        X = np.array([3, 4, 76, 4, 5])
        Y = np.array([0, 3, 4])

        msm_distance = msm(X, Y, c = 1, w = 5)
        print(msm_distance)

    Output:

    .. code-block:: bash

        79.0

    **References:**

    .. [1] Alexandra  Stefan,  Vassilis  Athitsos,  and  Gautam  Das.  “The  Move-Split-Merge Metric for Time Series”. In:IEEE Transactions on Knowledge andData Engineering25.1425 – 1438 (2013).
    """
    
    if constraint == "None":
        if fast == True:
            return msm_n_numba(x,y,c)
        if fast == False:
            return msm_n(x,y,c)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return msm_scb_numba(x,y,c,w)
        if fast == False:
            return msm_scb(x,y,c,w)

@jit(nopython=True)
def msm_dist(new, x, y, c):
    if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
        dist = c
    else:
        dist = c + min(abs(new - x), abs(new - y))
    return dist


@jit(nopython=True)
def msm_n_numba(x,y,c):
    xlen = len(x)
    ylen = len(y)
    cost = np.full((xlen, ylen), np.inf)
    cost[0][0] = abs(x[0] - y[0])
    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c)
    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c)
    for i in range(1,xlen):
        for j in range(0, ylen):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c))
    return cost[xlen-1][ylen-1]

def msm_n(x,y,c):
    xlen = len(x)
    ylen = len(y)
    cost = np.full((xlen, ylen), np.inf)
    cost[0][0] = abs(x[0] - y[0])
    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c)
    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c)
    for i in range(1,xlen):
        for j in range(0, ylen):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c))
    return cost[xlen-1][ylen-1]

@jit(nopython=True)
def msm_scb_numba(x,y,c,w):
    xlen = len(x)
    ylen = len(y)
    if w == None:
        w = max(xlen, ylen)
    cost = np.full((xlen, ylen), np.inf)
    cost[0][0] = abs(x[0] - y[0]);

    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c)

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c)

    for i in range(1,xlen):
        for j in range(max(0, int(i-w)), min(ylen, int(i+w))):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c))

    return cost[xlen-1][ylen-1]


def msm_scb(x,y,c,w):
    xlen = len(x)
    ylen = len(y)
    if w == None:
        w = max(xlen, ylen)
    cost = np.full((xlen, ylen), np.inf)

    cost[0][0] = abs(x[0] - y[0])

    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c)

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c)

    for i in range(1,xlen):
        for j in range(max(0, int(i-w)), min(ylen, int(i+w))):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c))

    return cost[xlen-1][ylen-1]

def lb_msm(y, x, c, w = None, fast = True):
    
    r"""LB_MSM is an MSM lower bound which constructs upper and lower envelopes, 
    formally defined as:

    .. math::

       
        \begin{equation}
            LB\_MSM(X, Y) = |X_1 - Y_1| + \overset{L_Y}{\underset{i=2}{\sum}}\begin{cases}
                                min(|Y_i - X_{max}|, c) \textup{ if } Y_{i-1}\geq{Y_{i}} > X_{max}   \\
                        min(|Y_i - X_{min}|, c) \textup{ if } Y_{i-1}\leq{Y_{i}} < X_{min} \\
                        0 \textup{ otherwise }
                            \end{cases}
        \end{equation}
        

    :param y: a time series
    :type y: np.array
    :param x: another time series
    :type x: np.array
    :param c: the cost for one *move* or *split** operation 
    :type c: float
    :param w: ``w`` is the largest temporal shift allowed between two time series, defaulting to ``None``
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation, defaulting to ``True``
    :type fast: bool, optional
    :return: LB_MSM distance

    **Example:**
    
    Input:

    .. code-block:: python

        from elastic import lb_msm
        import numpy as np

        X = np.array([-3, 2, -4, -5, -3, -2, -93])
        Y = np.array([3, 13, 3, 1, 6, 9, 9])
        lb_msm_dist = lb_msm(Y, X, c = 2, w = 3)
        print(lb_msm_dist)

    Output:

    .. code-block:: bash

      9.0


    **References**

    .. [1] Chang Wei Tan, François Petitjean, and Geoffrey I Webb. 2020. FastEE: Fast
           Ensembles of Elastic Distances for time series classification. Data Mining and
           Knowledge Discovery 34, 1 (2020), 231–272.
    """
    
    if fast == True:
        return lb_msm_numba(y, x, c, w)
    if fast == False:
        return lb_msm_n(y, x, c, w)


def lb_msm_n(y, x, c, w = None):

    lenx = len(x)
    leny = len(y)

    if w == None:
        w = max(lenx, leny)
    
    XUE, XLE = make_envelopes(x, w)
    
    lb_dist = abs(x[0]-y[0])

    for i in range(1,leny):

        if y[i] > XUE[i] and y[i-1] >= y[i]:
            lb_dist += min(abs(y[i]-XUE[i]), c)
        if y[i] < XLE[i] and y[i-1] <= y[i]:
            lb_dist += min(abs(y[i]-XLE[i]), c)
    
    return lb_dist


@jit(nopython = True)
def lb_msm_numba(y, x, c, w = None):

    lenx = len(x)
    leny = len(y)

    if w == None:
        w = max(lenx, leny)
    
    XUE, XLE = make_envelopes(x, w)
    
    lb_dist = abs(x[0]-y[0])

    for i in range(1,leny):

        if y[i] > XUE[i] and y[i-1] >= y[i]:
            lb_dist += min(abs(y[i]-XUE[i]), c)
        if y[i] < XLE[i] and y[i-1] <= y[i]:
            lb_dist += min(abs(y[i]-XLE[i]), c)
    
    return lb_dist

# End of MSM

# Start of Swale
def swale(x,y,p,r,epsilon,constraint=None,w=None, fast = True):

    r"""
    Sequence Weighted Alignment (SWALE) [1]_ is an :math:`\epsilon` based distance measure. 
    SWALE introduces a punishment and reward system that is not in Longest Common Subsequence (LCSS). 
    This is encapsulated the parameters p and r. 
    This allows the user to tailor how punishing mismatches are and how rewarding matches are. 
    This makes SWALE more detailing then LCSS as LCSS only records the number of matches 
    and Edit Distance on Real Sequences (EDR) as EDR does not rewards matches. 
    However, 
    this leaves the responsibility to the user to set three parameters to get meaningful results. 
    This can be very hard to do without extensive testing 
    and leaves the results of SWALE heavily variable to the parameters users choose.

    .. math:: 

        \begin{aligned}
            D^u(x,y,\epsilon,p,r) &= 0\\
            D^v(x,y,\epsilon,p,r) = D^h(x,y,\epsilon,p,r) & = 
            \begin{cases}
                \infty & \text{if $|x - y| \leq \epsilon$}\\
                p & \text{else}\\
            \end{cases}\\
            D^d(x,y,\epsilon,p,r)& =
            \begin{cases}
                r & \text{if $|x - y| \leq \epsilon$}\\
                \infty & \text{else}\\
            \end{cases}\\
            \pi(d_{n,m}) &= d_{n,m}
        \end{aligned}

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param p: punishment of one mismatch
    :type p: float
    :param r: reward of one match
    :type r: float
    :param epsilon: the matching threshold
    :type epsilon: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; ``w`` defaults to ``None``, in which case the warping window is the length of the longest time series.
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: SWALE distance

    **Example:**

    Input:

    .. code-block:: python

        X = np.array([3, 4, 76, 4, 5])
        Y = np.array([0, 3, 4])

        swale_distance = swale(X, Y, p = 1, r = 1, epsilon = 3)
        print(swale_distance)

    Output:

    .. code-block:: bash

        6.0

    **References:**

    .. [1] Michael D. Morse and Jignesh M. Patel. “An efficient and accurate methodfor  evaluating  time  series  similarity”.  In:Proceedings of the 2007 ACMSIGMOD international conference on Management of data569–580 (2007).
    """
    
    if constraint == "None":
        if fast == True:
            return swale_n_numba(x,y,p,r,epsilon)
        if fast == False:
            return swale_n(x,y,p,r,epsilon)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
           return swale_scb_numba(x,y,p,r,epsilon,w)
           
        if fast == False:
            return swale_scb(x,y,p,r,epsilon,w)

@jit(nopython=True)
def swale_n_numba(x,y,p,r,epsilon):
    cur = np.zeros(len(y))
    prev = np.zeros(len(y))
    for i in range(len(x)):
        prev = cur
        cur = np.zeros(len(y))
        minw = 0
        maxw = len(y)-1
        for j in range(int(minw),int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = j * p
            elif j == minw:
                cur[j] = i * p
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r
                else:
                   cur[j] = min(prev[j], cur[j-1]) + p
    return cur[len(y)-1]

def swale_n(x,y,p,r,epsilon):
    cur = np.zeros(len(y))
    prev = np.zeros(len(y))
    for i in range(len(x)):
        prev = cur
        cur = np.zeros(len(y))
        minw = 0
        maxw = len(y)-1
        for j in range(int(minw),int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = j * p
            elif j == minw:
                cur[j] = i * p
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r
                else:
                   cur[j] = min(prev[j], cur[j-1]) + p
    return cur[len(y)-1]

@jit(nopython=True)
def swale_scb_numba(x,y,p,r,epsilon,w):
    lenx = len(x)
    leny = len(y)
    if w == None:
        w = max(lenx, leny)
    cur = np.zeros(leny)
    prev = np.zeros(leny)
    if w == None:
        w = max()
    for i in range(lenx):
        prev = cur
        cur = np.zeros(leny)
        minw = max(0,i-w)
        maxw = min(i+w,leny-1)
        for j in range(int(minw),int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = j * p
            elif j == minw:
                cur[j] = i * p
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r
                else:
                   cur[j] = min(prev[j], cur[j-1]) + p
    return cur[leny-1]

def swale_scb(x,y,p,r,epsilon,w):
    lenx = len(x)
    leny = len(y)
    if w == None:
        w = max(lenx, leny)
    cur = np.zeros(leny)
    prev = np.zeros(leny)
    if w == None:
        w = max()
    for i in range(lenx):
        prev = cur
        cur = np.zeros(leny)
        minw = max(0,i-w)
        maxw = min(i+w,leny-1)
        for j in range(int(minw),int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = j * p
            elif j == minw:
                cur[j] = i * p
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r
                else:
                   cur[j] = min(prev[j], cur[j-1]) + p
    return cur[leny-1]

# End of Swale