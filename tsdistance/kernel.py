import numpy as np
import math
from scipy.spatial.distance import cdist

def kdtw_distance(x, y, sigma):
    factor = 1.0/3
    minprob = 10 ** (-20)
    return factor * (math.exp(-sigma * ((x-y)**2))+minprob)

def kdtw(x, xlen, y, ylen, sigma):

    r"""
    Kernel Dynamic Time Warping (KDTW) [1]_ is a similarity measure constructed from DTW 
    with the property that KDTW is a positive definite kernel 
    (homogeneous to an inner product in the so-called Reproducing Kernel Hilbert Space). 
    Following earlier work by Cuturi & al. [2]_, 
    namely the so-called Global Alignment kernel (GA-kernel), 
    the derivation of KDTW is detailed in Marteau & Gibet 2014  [1]_. 
    KDTW is a convolution kernel as defined in [3]_. The formula for KDTW is shown below:

    .. math::

        \begin{equation*}
            k(X_i,Y_i,\sigma) = e^{- (X_i-Y_i)^2/\sigma }
        \end{equation*}
    
    .. math::

        \begin{equation*}
            KDTW^{xy}(X_i,Y_j,\sigma) = \beta * k(X_i,Y_j,\sigma) \cdot \sum
            \begin{cases}
                h(i-1,j)KDTW^{xy}(X_{i-1},Y_j) \\ h(i-1,j-1)KDTW^{xy}(X_{i-1},Y_{j-1}) \\ h(i,j-1)KDTW^{xy}(X_i,Y_{j-1}) \\
            \end{cases}
        \end{equation*}
    
    .. math::

        \begin{equation*}
            KDTW^{xx}(X_i,Y_j,\sigma) = \beta \cdot \sum 
            \begin{cases}
            (h(i-1,j) KDTW^{xx}(X_{i-1},Y_j) * k(X_{i},Y_i,\sigma) \\ \Delta_{i,j} *h(i,j)*KDTW^{xx}(X_{i-1},Y_{j-1})*k(x_i,y_j,\sigma) \\ h(p,q-1)*KDTW^{xx}(X_i,Y_{j-1})*k(X_j,Y_j,\sigma) \\
            \end{cases}
        \end{equation*}
    
    .. math::

        \begin{equation*}
            KDTW(X,Y) = KDTW^{xy}(X_n,Y_m) + KDTW^{xx}(X_n,Y_m) 
        \end{equation*}
    
    :param x: time series :code:`x`
    :type x: np.array
    :param xlen: length of time series :code:`x`
    :type xlen: int
    :param y: time series :code:`x`
    :type y: np.array
    :param ylen: length of time series :code:`y`
    :type ylen: int
    :param sigma: bandwidth parameter which weights the local contributions
    :type sigma: float
    :return: the KDTW distance

    **Reference**

    .. [1] Pierre-François Marteau and Sylvie Gibet. “On Recursive Edit DistanceKernels with Application to Time Series Classification”. In:IEEE Trans-actions on Neural Networks and Learning Systems1-14 (2014)
    
    .. [2] M. Cuturi et al. “A Kernel for Time Series Based on Global Alignments”.In:IEEE International Conference on Acoustics, Speech and Signal Pro-cessing2.413-416 (2007)
    
    .. [3] David Haussler. “Convolution Kernels on Discrete Structures”. In:Techni-cal Report UCS-CRL-99-10, University of California at Santa Cruz, SantaCruz, CA, USA.(1999)
    
    """

    xp = np.array([])
    yp = np.array([])

    xp[0]=0
    yp[0]=0

    for i in range(1, xlen+1):
        xp[i] = x[i-1]
    for i in range(1, ylen+1):
        yp[i] = y[i-1]

    xlen = xlen + 1
    ylen = ylen + 1

    x = xp
    y = yp

    dp = np.array([])
    dp1 = np.array([])
    dp2 = np.array([])

    dp2[0] = 1

    for i in range(1, min(xlen, ylen)):
        dp2[i] = kdtw_distance(x[i], y[i], sigma)
    
    for i in range(0, xlen):
        dp[i] = np.array([])
        dp1[i] = np.array([])

    len = min(xlen, ylen)

    dp[0][0] = 1
    dp1[0][0] = 1

    for i in range(1, xlen):
        dp[i][0] = dp[i - 1][0] * kdtw_distance(x[i], y[1], sigma)
        dp1[i][0] = dp1[i - 1][0] * dp2[i]

    for i in range(1, ylen):
        dp[0][i] = dp[0][i - 1] * kdtw_distance(x[1], y[i], sigma)
        dp1[0][i] = dp1[0][i - 1] * dp2[i]
    
    for i in range(1, xlen):
        for j in range(1, ylen):
             lcost = kdtw_distance(x[i], y[j], sigma)
             dp[i][j] = (dp[i - 1][j] + dp[i][j - 1] + dp[i - 1][j - 1]) * lcost
             if i ==j:
                 dp1[i][j] = dp1[i - 1][j - 1] * lcost + dp1[i - 1][j] * dp2[i] + dp1[i][j - 1] * dp2[j]
             else:
                dp1[i][j] = dp1[i - 1][j] * dp2[i] + dp1[i][j - 1] * dp2[j]
    
    for i in range(0, xlen):
        for j in range(0, ylen):
            dp[i][j] += dp1[i][j]
    
    ans = dp[xlen - 1][ylen - 1]

    return ans


def nextpow2(N):
    n = 1
    while n < N: n *= 2
    return n

def PreservedEnergy(x, e):
    FFTx = np.fft.fft(x, 2**(nextpow2(2*len(x)-1))) 
    NormCumSum = np.cumsum(math.abs(FFTx**2))/np.sum(math.abs(FFTx**2))
    k = np.argwhere(NormCumSum >= (e /2))[0]
    FFTx[k+1:len(FFTx)-k-1] = 0 
    return FFTx


def NCC(x, y, e):
    FFTx = PreservedEnergy(x, e)
    FFTy = PreservedEnergy(y, e)
    return np.fft.ifft(FFTx * FFTy)/np.dot(np.linalg.norm(x),np.linalg.norm(y))

def SumNCC(x, y, gamma, e):
    return np.sum(math.exp(gamma * NCC(x, y, e)))


def SINK(x, y, gamma, e):
    r"""
    Shift Invariant Kernel (SINK) [1]_ [2]_
    computes the distance between time series X and Y by summing all weighted elements of the Coefficient Normalized Cross-Correlation 
    (:math:`NCC_c`) sequence between :math:`X` and :math:`Y`. 
    Formally, SINK is defined as follows:

    .. math::

        \begin{equation}
            SINK(x,y,\gamma) = \sum_{i=1}^ne^{\gamma * NCCc_i(x,y)}
        \end{equation} 
    
    where :math:`\gamma > 0`.

    :param x: time series :code:`x`
    :type x: np.array
    :param y: time series :code:`x`
    :type y: np.array
    :param gamma: bandwidth paramater that determines weights for each inner product through :math:`k'(\vec{x}, \vec{y}, \gamma) = e^{\gamma<\vec{x}, \vec{y}>}`
    :type: float, :math:`\gamma` > 0
    :param e: constant, default to :math:`e`
    :return: the SINK distance

    **References**

    .. [1] John Paparrizos and Michael Franklin. “GRAIL: Efficient Time-SeriesRepresentation Learning”. In:Proceedings of the VLDB Endowment12(2019)

    .. [2] Amaia Abanda, Usue Mor, and Jose A. Lozano. “A review on distancebased time series classification”. In:Data Mining and Knowledge Discovery12.378–412 (2019)
    
    """

    return SumNCC(x, y, gamma, e) / math.sqrt(SumNCC(x, x, gamma, e)*SumNCC(x, x, gamma, e))


def LGAK(x, y, sigma):
    r"""
    This function uses the log Global Alignment Kernel (TGAK) described in Cuturi (2011) [1]_.
    The formula for LGAK is follows:

    .. math::

        LGAK(x, y,\sigma)= (\prod_{i=1}^{|\pi|}e^(\frac{1}{2\sigma^2}({x_{\pi_1(i)} - y_{\pi_2(j)}})^2+log(e^{-\frac{({x_{\pi_1(i)} - y_{\pi_2(j)}})^2}{2\sigma^2}})))
    
    :param x: time series :code:`x`
    :type x: np.array
    :param y: time series :code:`x`
    :type y: np.array
    :param sigma: parameter of the Gaussian kernel
    :type sigma: float
    :return: the LGAK distance

    """

    K = np.exp(-(cdist(x, y, "sqeuclidean") / (2 * sigma ** 2) + np.log(2 - np.exp(cdist(x, y, "sqeuclidean") / (2 * sigma ** 2)))))

    csum = np.zeros((len(x)+1, len(y)+1))
    csum[0][0] = 1
    for i in range(len(x)):
        for j in range(len(y)):
            csum[i+1][j+1] = (csum[i, j + 1] + csum[i + 1, j] + csum[i, j]) * K[i][j]

    return csum[len(x)][len(y)]

    

    
