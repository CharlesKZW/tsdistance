import numpy as np
import math



def minkowski(x,y,p):

    r"""
    The formula for minkowski function is: :math:`\begin{equation*}(\sum_{i=1}^n |X_i - Y_i|^p)^{\frac{1}{p}}\end{equation*}`
    
    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :param p: parameter for :math:`p` in the formula above
    :type p: float
    :return: the Minkowski distance
    """

    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]) ** p;

    return sum ** (1/p);

def abs_euclidean(x,y):

    r"""
    Euclidean distance is our most intuitive way of defining distance as that's how we define it in our physical world.
    The formula is: :math:`\sqrt{\sum_{i=1}^n(X_i - Y_i)^2}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Euclidean distance

    """

    if len(x) != len(y):
        return -1;
    sum = 0
    for i in range(len(x)):
        sum += abs(x[i] - y[i]) ** 2;

    return sum ** (1/2);


def manhattan(x,y):

    r"""
    Manhattan distance is when :math:`p = 1`. 
    Manhattan distance is often called city-block distance as in the 2-dimensional case it is often represented using city-blocks. 
    Manhattan distance's advantage is that outliers skew the result less than in Chebyshev or Euclidean distance.
    The formula is: :math:`\sum_{i=1}^n |X_i - Y_i|`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Manhattan distance
    """

    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum+= (abs(x[i] - y[i]));
    return sum;


def chebyshev(x,y):

    r"""
    {Chebyshev distance} is represented as the limit as p tends towards infinity. 
    Chebyshev distance is computed as: :math:`max_i(X_i - Y_i) = \lim_{p \rightarrow \infty} (\sum_{i=1}^n |X_i - Y_i|^p)^{\frac{1}{p}}`
    
    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Chebyshev distance
    """

    if len(x) != len(y):
        return -1;
    max = 0
    for i in range(len(x)):
        dif = abs(x[i] - y[i]);
        if max < dif:
            max = dif;
    return max;


def sorensen(x,y):
    r"""
    Sorensen distance is the :math:`L_1` distance but divided by the sum of the two time series. 
    Because of this, the range of the Sorensen distance is :math:`[0,1]`. 
    It is often used in ecology and environmental sciences.
    The formula is: :math:`\frac{\sum_{i=1}^n |X_i - Y_i|}{\sum_{i=1}^n(X_i + Y_i)}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Sorensen distance

    """
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += abs(x[i] - y[i]);
        sumb += (x[i] + y[i]);
    return suma/sumb;

def gower(x,y):
    r"""
    Gower distance is the average distance between the elements. 
    It is often used for mixed qualitative and quantitative data.
    The formula is: :math:`\frac{1}{n} * \sum_{i=1}^n |X_i - Y_i|`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Gower distance

    """
    if len(x) != len(y):
        return -1;
    if len(x) == 0:
        return -1;
    sum = 0;
    for  i in range(len(x)):
        sum += abs(x[i] - y[i]);
    return 1/len(x) * sum;


def soergel(x,y):
    r"""
    Soergel distance is the :math:`L_1` distance divided by the sum of the maximum of each element pair.
    The formula is: :math:`\frac{\sum_{i=1}^n |X_i - Y_i|}{\sum_{i=1}^n max(X_i,Y_i)}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Soergel distance

    """
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += abs(x[i] - y[i])
        sumb += max((x[i],y[i]));
    return suma/sumb;

def Kulczynski(x,y):
    r"""

    Kulczynski distance is very similar but the :math:`L_1` distance is divided by the sum of the minimum of each element pair.
    The formula is: :math:`\frac{\sum_{i=1}^n|X_i - Y_i|}{\sum_{i=1}^n min(X_i,Y_i)}`.

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Kulczynski distance

    """
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += abs(x[i] - y[i]);
        sumb += min(x[i], y[i]);
    if sumb == 0:
        return -1;
    return suma/sumb;

def canberra(x,y):
    r"""
    Canberra distance is the :math:`L_1` distance but each element difference is divided by the element sum. 
    Canberra distance is often used for data scattered about an origin.
    The formula is: :math:`\sum \frac{|X_i - Y_i|}{X_i + Y_i}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Canberra distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] + y[i] == 0:
            return -1;

        sum += abs(x[i] - y[i]) / (x[i] + y[i]);

    return sum;

def lorentzian(x,y):
    r"""
    Lorentzian distance is the natural log of the :math:`L_1` distance between to time series.
    To avoid :math:`ln(0)` and guarantee non-negative distances 1 is added.
    The formula is: :math:`sum_{i=1}^n ln(1 + |X_i - Y_i|)`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Lorentzian distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += math.log(1 + abs(x[i] - y[i]))
    return sum;

def Intersection(x,y):
    r"""
    Intersection distance is the :math:`L_1` distance divided by 2.
    The formula is: :math:`\frac{\sum_{i=1}^n |X_i - Y_i|)}{2}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Intersection distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]);
    
    return 1/2 * sum;

def wave_hedges(x,y):
    r"""
    Wave Hedges distance is the length of the time series subtracted by the sum of the ratio of the minimum and maximum of each element pair.
    The formula is: :math:`\sum_{i=1}^n1 - \frac{min(X_i,Y_i)}{max(X_i,Y_i)}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Wave Hedges distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += min((x[i],y[i])) / max((x[i],y[i]));
        
    return len(x) - sum


def czekanowski(x,y):
    r"""
    Czekanowski distance is the intersection equivalent of Sorensen. 
    It is the sum of the minimums of each element pair divided by the sum of the elements multiplied by 2.
    The formula is: :math:`2\frac{\sum_{i=1}^nmin(X_i,Y_i)}{\sum_{i=1}^nX_i + Y_i}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Czekanowski distance
    """

    if len(x) != len(y):
        return -1;
    sum_add = 0;
    sum_dif = 0;
    for i in range(len(x)):
        sum_add += (x[i] + y[i]);
        sum_dif += abs(x[i] - y[i]);

    if sum_add == 0:
        return -1;

    return sum_dif/sum_add;

def motyka(x,y):
    r"""
    Motyka distance is the sum of the minimums of each element pair divided by the sum of the elements of each time series.
    The formula is: :math:`\frac{\sum_{i=1}^nmin(X_i,Y_i)}{\sum_{i=1}^nX_i + Y_i}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Motyka distance
    """
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += max((x[i],y[i]))
        sumb += x[i] + y[i];
    return suma/sumb;

def tanimoto(x,y):
    r"""
    Tanimoto distance is equivalent to the Soergel distance measure. 
    It is the difference between the maximum and minimum of each element pair divided by the maximums of each element pair.
    The formula is: :math:`\frac{\sum_{i=1}max(X_i,Y_i) - min(X_i,Y_i)}{\sum_{i=1}^nmax(X_i,Y_i)}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Tanimoto distance
    """
    if len(x) != len(y):
        return -1;
    minxy = np.minimum(x,y);
    sumxy = np.sum(x) + np.sum(y);
    a = (sumxy - 2 * minxy)
    b = np.linalg.pinv([sumxy - minxy])
    return np.sum(np.dot(a,b));

def innerproduct(x,y):
    r"""
    Inner Product distance is the dot product between two time series.
    The formula is: :math:`\sum_{i=1}^n (X_iY_i)`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Inner Product distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += (x[i] * y[i]);

    return sum;

def harmonicmean(x,y):
    r"""
    Harmonic Mean distance is the sum of the element-wise harmonic means between the time series. 
    It is often used when discussing rates of change.
    The formula is: :math:`2\sum_{i=1}^n (\frac{X_iY_i}{X_i + Y_i})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Harmonic Mean distance
    """
    if len(x) != len(y):
        return -1;

    a = np.multiply(x,y);
    b = np.linalg.pinv([np.add(x,y)]);

    return 2 * np.sum(np.dot(a,b))

def kumarhassebrook(x,y):
    r"""
    Kumar-Hassebrook distance is like harmonic mean distance but the denominator is reduced by the product of the elements.
    The formula is: :math:`\frac{\sum_{i=1}^nX_iY_i}{\sum_{i=1}^n(X_i + Y_i) - \sum_{i=1}^n X_iY_i}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Kumar-Hassebrook distance
    """
    if len(x) != len(y):
        return -1;
    return np.sum(np.multiply(x,y)) /np.subtract(np.add(np.sum(np.square(x)),np.sum(np.square(y))),np.sum(np.multiply(x,y)))

def jaccard(x,y):
    r"""
    Jaccard distance is a metric and the complement of the Jaccard similarity coefficient.
    The formula is: :math:`\frac{\sum_{i=1}^n(X_i - Y_i)^2}{\sum_{i=1}^n (X^2 + y^2) - \sum_{i=1}^n (X_iY_i)}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Jaccard distance     
    """
    if len(x) != len(y):
        return -1;
    return np.sum(np.square(np.subtract(x,y))) / np.sum(np.square(x) + np.square(y) + np.multiply(np.add(x,y),-1));

def cosine(x,y):
    r"""
    Cosine distance is the complement of the cosine similarity that measures the angle between two vectors.
    As compared to the Inner Product distance, Cosine distance does not take the time series magnitude into account.
    The formula is: :math:`1 - \frac{\sum_{i=1}^n X_iY_i}{\sqrt{\sum_{i=1}^nX_i^2}\sqrt{\sum{i=1}^nY_i^2}}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Cosine distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    sumx = 0
    sumy = 0
    for i in range(len(x)):
        sumy += y[i] ** 2;
        sumx += x[i] ** 2;
        sum += x[i] * y[i];
    if sumx < 0:
        return -1;
    if sumy < 0:
        return -1;

    return 1 - (sum/ ((sumx ** (1/2))*(sumy ** (1/2))));

def dice(x,y):
    r"""
    Dice distance is the complement of the Dice similarity. 
    It is not a metric but it is widely used in biological taxonomy.
    The formula is: :math:`1 - \frac{2\sum_{i=1}^nX_iY_i}{\sum_{i=1}^nX^2 + y^2}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Dice distance
    """
    if len(x) != len(y):
        return -1;
    sum_dif = 0;
    sum_add = 0;
    for i in range(len(x)):
        sum_dif += (x[i] - y[i]) ** 2;
        sum_add += (x[i] ** 2 + y[i] ** 2);

    if (sum_add == 0):
        return -1;
    return sum_dif/sum_add;

def fidelity(x,y):
    r"""
    Fidelity distance is the sum of the square root of the element-wise product of elements from two time series. 
    The formula is: :math:`\sum_{i = 1}^n \sqrt{X_iY_i}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Fidelity distance

    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum += (x[i] * y[i]) ** (1/2);

    return sum;

def bhattacharyya(x,y):
    r"""
    Bhattacharyya distance is a statistal distance metric that measures the similarity of two probability distributions. 
    It is the general case of Mahalanobis distance.
    The formula is: :math:`-ln(\sum_{i=1}^n\sqrt{X_iY_i})`
    
    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Bhattacharyya distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum += math.sqrt(x[i] * y[i]);

    return - math.log(sum);

def Square_chord(x,y):
    r"""
    Squared Chord distance is the sum of the square of the differences of the square roots of each element. 
    This exaggerates more dissimilar features.
    The formula is: :math:`\sum_{i=1}^n(\sqrt{X_i}-\sqrt{Y_i})^2`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Squared Chord distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] ** (1/2)) - (y[i]) ** (1/2)) ** 2;

    return sum;

def hellinger(x,y):
    r"""
    Hellinger Distance is Matusita distanced scaled by :math:`\sqrt{2}`.
    The formula is: :math:`\sqrt{2\sum_{i=1}^n(\sqrt{X_i}-\sqrt{Y_i})^2}`.

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Hellinger Distance
    """
    if len(x) != len(y):
        return -1;
    sum = 1;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum -= (x[i] * y[i]) ** (1/2);

    if (sum < 0):
        return -1
    return 2 * (sum ** (1/2))

def matusita(x,y):
    r"""
    Matusita Distance is the square root of the squared chord distance.
    The formula is: :math:`\sqrt{\sum_{i=1}^n(\sqrt{X_i}-\sqrt{Y_i})^2}`.

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Matusita Distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] * y[i] < 0:
            return -1;
        sum += (x[i] * y[i]) ** (1/2);

    result = 2 - 2 * sum;
    if result < 0:
        return -1;
    return result ** (1/2);

def squared_euclidean(x,y):

    r"""
    Squared Euclidean distance is the square of the Euclidean distance.
    The formula is: :math:`\sum_{i=1}^n (X_i - Y_i)^2`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Squared Euclidean distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum+= (x[i] - y[i]) ** 2;
    return sum;

def clark(x,y):
    r"""
    Clark distance is the square root of the sum of the squared ratio of the difference and sum of the element pairs.
    The formula is: :math:`\sqrt{\sum_{i=1}^n(\frac{|X_i - Y_i|}{X_i + Y_i})^2}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Clark distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i] + y[i] == 0:
            return -1;
        sum += (abs(x[i] - y[i]) ** 2) / (x[i] + y[i])

    return sum ** (1/2);

def neyman(x,y):
    r"""
    Neyman Chi Squared distance is the sum of squared difference of the element pairs divided by the element in the first time series.
    The formula is :math:`\sum_{i=1}^n(\frac{(X_i - Y_i)^2}{X_i})`.

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Neyman Chi Squared distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2 / x[i];

    return sum;

def pearson(x,y):
    r"""
    Pearson Chi Squared distance is the sum of squared difference of the element pairs divided by the element in the second time series. 
    Notably, :math:`Pearson(X,Y)` is equal to :math:`Neyman(Y,X)`. 
    The formula is: :math:`\sum_{i=1}^n(\frac{(X_i - Y_i)^2}{Y_i})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Pearson Chi Squared distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2) / y[i]

    return sum;

def squared_chi(x,y):
    r"""
    Squared Chi distance is the sum of the squared difference of the element pairs divided by the sum of the element pairs. 
    This can be considered a symmetric version of the Neyman Chi Squared distance.
    The formula is: :math:`\sum_{i=1}^n(\frac{(X_i - Y_i)^2}{X_i + Y_i}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Pearson Squared Chi distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2)/ (x[i] + y[i]);
    return sum;

def K_divergence(x,y):
    r"""
    Divergence distance is the sum of the squared difference of the element pairs over the squared sum multplied by 2. 
    Divergence distance is not a metric.
    The formula is: :math:`2\sum_{i=1}^n\frac{(X_i - Y_i)^2}{(X_i + Y_i)^2}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Divergence distance
    """
    if len(x) != len(y):
        return -1;
    for i in range(len(x)):
        if y[i] <= 0:
            return -1;
    return np.sum(np.multiply(x,np.log(np.divide(np.multiply(x,2),np.add(x,y)))));


def additive_symm_chi(x,y):
    
    r"""
    Additive Symmetric Chi distance is the sum of the square of the difference of the element pairs multiplied by the sum of the element pairs divided by the product of the element pairs.
    The formula is: :math:`2\sum_{i=1}^n\frac{(X_i - Y_i)^2(X_i + Y_i)}{X_iY_i}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Divergence distance
    """

    if len(x) != len(y):
        return -1;

    sum = 0;
    for i in range(len(x)):
        if x[i] == 0:
            return -1;
        if y[i] == 0:
            return -1;
        sum += (x[i] - y[i]) ** 2 * (x[i] + y[i]) / (x[i] * y[i]);

    return sum;

def prob_symmetric_chi(x,y):
    r"""
    Probabilistic Symmetric Chi distance is Squared Chi distance multiplied by 2.
    The formula is: :math:`2\sum_{i=1}^n\frac{(X_i - Y_i)^2}{X_i + Y_i}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Probabilistic Symmetric Chi distance

    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2) / (x[i] + y[i]);

    return 2 * sum;

def kullback(x,y):
    r"""
    Kullback-Leibler distance is known as KL divergence or information deviation. 
    It is a measure of how different two probability distributions are to each other.
    The formula is: :math:`\sum_{i=1}^nX_iln(\frac{2X_i}{X_i + Y_i})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Kullback-Leibler distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        if x[i]/y[i] == 0:
            return -1;
        sum+= x[i] * math.log(x[i]/y[i]);

    return sum;


def jeffrey(x,y):
    r"""
    Jeffreys distance is considered to be the symmetric version of Kullback-Leibler distance. 
    The formula is: :math:`\sum_{i=1}^n(X_i-Y_i)ln(\frac{X_i}{Y_i})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Jeffreys distance
    """

    if len(x) != len(y):
        return -1;
    return np.sum(np.multiply((np.subtract(x,y)),np.log(np.divide(x,y))));

def K_divergence(x,y):
    r"""
    Divergence distance is the sum of the squared difference of the element pairs over the squared sum multplied by 2. 
    Divergence distance is not a metric.
    The formula is: :math:`2\sum_{i=1}^n\frac{(X_i - Y_i)^2}{(X_i + Y_i)^2}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Divergence distance
    """
    if len(x) != len(y):
        return -1;
    for i in range(len(x)):
        if y[i] <= 0:
            return -1;
    return np.sum(np.multiply(x,np.log(np.divide(np.multiply(x,2),np.add(x,y)))));


def topsoe(x,y):
    r"""
    Topsoe distance is a symmetric version of K divergence distance.
    The formula is: :math:`\sum_{i=1}^nX_iln(\frac{2X_i}{X_i + Y_i}) + Y_iln(\frac{2Y_i}{Y_i + X_i})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Topsoe distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    logxy = []
    for i in range(len(x)):
        logxy.append(math.log(x[i] + y[i]));
    for i in range(len(x)):
        sum += (x[i] * (math.log(2*x[i]) - logxy[i])) + (y[i] * (math.log(2*y[i]) - logxy[i]));
    return sum;


def jensen_shannon(x,y):
    r"""
    Jensen-Shannon distance is Topsoe distance divided by 2.
    The formula is: :math:`\frac{\sum_{i=1}^nX_iln(\frac{2X_i}{X_i + Y_i}) + Y_iln(\frac{2Y_i}{Y_i + X_i})}{2}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Jensen-Shannon distance
    """
    if len(x) != len(y):
        return -1;
    logxy = [];
    for i in range(len(x)):
        if x[i] + y[i] <= 0:
            return -1;
        logxy.append(math.log(x[i] + y[i]));

    sum = 0
    for i in range(len(x)):
        if x[i] <= 0:
            return -1;
        if y[i] <= 0:
            return -1;
        sum += x[i] * (math.log(2*x[i]) - logxy[i]) + y[i] * (math.log(2*y[i]) - logxy[i])
    
    return .5 * sum;

def jensen_difference(x,y):
    r"""
    The formula for Jensen Difference Distance is: :math:`\sum_{i=1}^n \frac{X_iln(X_i) + Y_iln(Y_i)}{2} - \frac{X_i + Y_i}{2} * ln(\frac{X_i + Y_i}{2})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Jensen Difference Distance
    """
    xyavg = [];
    for i in range(len(x)):
        if x[i] + y[i] <= 0:
            return -1;
        xyavg.append((x[i] + y[i])/2);

    sum = 0;
    for i in range(len(x)):
        if y[i] <= 0:
            return -1;
        if x[i] <= 0:
            return -1;
        sum += (x[i] * math.log(x[i]) + y[i] * math.log(y[i])) / 2 - xyavg[i] * math.log(xyavg[i]);

    return sum;



def vicis_wave_hedges(x,y):
    r"""
    Vicis-Wave Hedges distance is a variant of the Wave Hedges function and can be considered a :math:`L_1` function.
    The formula is: :math:`\sum_{i=1}^n \frac{X_i - Y_i}{min(X_i,Y_i)}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Vicis-Wave Hedges distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += abs(x[i] - y[i]) / min((x[i],y[i]))
        
    return sum;


def emanon2(x,y):
    r"""
    Emamon 2 distance is a variant of Vicis Wave Hedges where the squared differences and minimums are added together.
    The formula is: :math:`\sum_{i=1}^n \frac{(X_i - Y_i)^2}{min(X_i,Y_i)^2}`
    
    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Emamon 2 distance 
    """
    if len(x) != len(y):
        return -1;
    sum = 0
    comp = 0;
    for i in range(len(x)):
        mind = min(x[i],y[i]);
        if mind == 0:
            return -1;
        sum += ((x[i] - y[i]) ** 2) / mind ** 2;
    return sum;

def emanon3(x,y):
    r"""
    Emamon 3 distance is another variant of Vicis Wave Hedges where only the differences are squared.
    The formula is: :math:`\sum_{i=1}^n \frac{(X_i - Y_i)^2}{min(X_i,Y_i)}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Emamon 3 distance 
    """
    if len(x) != len(y):
        return -1;
    sum = 0
    comp = 0;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2) / min(x[i],y[i]);
    return sum;

def emanon4(x,y):
    r"""
    Emamon 4 distance is the last Emamon measure. It is the sum of the squared difference over the maximum of the element pairs.
    The formula is: :math:`\sum_{i=1}^n \frac{(X_i - Y_i)^2}{max(X_i,Y_i)}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Emamon 4 distance 
    """
    if len(x) != len(y):
        return -1;
    sum = 0
    comp = 0;
    maxd = max(x + y);
    if maxd == 0:
        return -1;
    for i in range(len(x)):
        sum += ((x[i] - y[i]) ** 2) / maxd;
    return sum;


def max_symmetric_chi(x,y):
    r"""
    Max-Symmetric Chi distance takes the maximum of the Pearson and Neyman distances.
    The formula is: :math:`max(\sum_{i=1}^n\frac{(X_i - Y_i)^2}{X_i},\sum_{i=1}^n\frac{(X_i - Y_i)^2}{Y_i})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Max-Symmetric Chi distance 
    """
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    xy = [];
    for i in range(len(x)):
        xy.append((x[i] - y[i]) ** 2);
    for i in range(len(x)):
        suma += (xy[i]/y[i]);
        sumb += (xy[i]/x[i]);
    return max((suma,sumb));

def min_symmetric_chi(x,y):
    r"""
    Min-Symmetric Chi takes the minimum of the Perason and Neyman distances.
    The formula is: :math:`min(\sum_{i=1}^n\frac{(X_i - Y_i)^2}{X_i},\sum_{i=1}^n\frac{(X_i-Y_i)^2}{Y_i})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Min-Symmetric Chi distance 
    """
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    xy = [];
    for i in range(len(x)):
        xy.append((x[i] - y[i]) ** 2);
    for i in range(len(x)):
        suma += (xy[i]/y[i]);
        sumb += (xy[i]/x[i]);
    return min((suma,sumb));


def taneja(x,y):
    r"""
    Taneja distance utilizes both the arithmetic and geometric mean.
    The formula is: :math:`\sum_{i=1}^n\frac{(X_i + Y_i)}{2} * ln(\frac{X_i + Y_i}{2\sqrt{X_iY_i}})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Taneja distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    xy = [];
    for i in range(len(x)):
        xy.append((x[i] + y[i])/ 2);
    
    for i in range(len(x)):
        sum += xy[i] * math.log(xy[i] / math.sqrt(x[i] *y[i]))

    return sum;


def kumar_johnson(x,y):
    r"""
    The formula for Kumar-Johnson distance is: :math:`\sum_{i=1}^n\frac{(X_i^2 - Y_i^2)^2}{2(X_iY_i)^{\frac{1}{2}}}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Kumar-Johnson distance
    """

    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] ** 2 - y[i] ** 2) ** 2) / (2 * (x[i] * y[i]) ** (3/2));
    return sum;


def avg_l1_linf(x,y):
    r"""
    Avg(:math:`L_1`,:math:`L_\infty`) is the average between the :math:`L_1` distance and Chebyshev distance.
    The formula is: :math:`\frac{\sum_{i=1}^n(|X_i - Y_i|) + max(X_i - Y_i)}{2}`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the Avg(:math:`L_1`,:math:`L_\infty`}) distance
    """
    if len(x) != len(y):
        return -1;
    sum = 0;
    max = 0;
    for i in range(len(x)):
        dif = abs(x[i] - y[i]);
        sum += dif;
        if (dif > max):
            max = dif;

    return (sum + max)/2;



def ED(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2;


    if sum < 0:
        return -1;

    return sum ** (1/2);







def jansen_shannon(x,y):
    if len(x) != len(y):
        return -1;
    logxy = [];
    for i in range(len(x)):
        if x[i] + y[i] <= 0:
            return -1;
        logxy.append(math.log(x[i] + y[i]));

    sum = 0
    for i in range(len(x)):
        if x[i] <= 0:
            return -1;
        if y[i] <= 0:
            return -1;
        sum += x[i] * (math.log(2*x[i]) - logxy[i]) + y[i] * (math.log(2*y[i]) - logxy[i])
    
    return .5 * sum;







def kulczynski(x,y):
    if len(x) != len(y):
        return -1;
    suma = 0;
    sumb = 0;
    for i in range(len(x)):
        suma += abs(x[i] - y[i]);
        sumb += min(x[i], y[i]);
    if sumb == 0:
        return -1;
    return suma/sumb;





def PairWiseScalingDistance(x,y):
    if len(x) != len(y):
        return -1;
    xy = [];
    for i in range(len(x)):
        xy.append(x[i] - y[i]);

    sumx = 0;
    sumxy = 0;
    for i in range(len(x)):
        sumx += x[i] ** 2;
        sumxy += xy[i] ** 2;

    return (sumxy ** (1/2))/(sumx ** (1/2));


def square_chord(x,y):
    if len(x) != len(y):
        return -1;
    sum = 0;
    for i in range(len(x)):
        sum += ((x[i] ** (1/2)) - (y[i]) ** (1/2)) ** 2;

    return sum;





