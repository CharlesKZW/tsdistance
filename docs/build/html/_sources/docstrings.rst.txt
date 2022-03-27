
Time series classification tasks usually rely on Nearest Neighbors Search, 
which classifies query series by computing their distances to all series in the training set,
and label query series with the same class as their nearest "neighbor". 
This operation is expensive 
because elastic measures used for each comparison is in :math:`\O_{L^2}` 
where :math:`\L` is the length of one series. 
To speed up Nearest Neighbors Search, 
Lower Bounding Measures first compute a *lower bound* of the similarity between two series 
with :math:`\O_{L}` complexity and go on to compute the actual :math:`\O_{L^2}`  similarity only when 
the *lower bound* is less than the distance.

The algorithm for computing Lower Bounding Longest Common Subsequence (LCSS) is:

.. code-block:: python

    def lbLCSS(x,y, epsilon): 
    
        xmin = np.subtract(x,epsilon);
        xmax = np.add(x,epsilon);
        leny = len(y)
        match = 0
        
        for i in range(leny):

            mind = min(xmin)
            maxd = max(xmax)

            if y[i] >= mind and y[i] <= maxd:
                match = match + 1;

         return 1 - (match/(min(len(x),len(y))))

**Input Arguments: **

    - metric (str): elastic measure to compute similarity, shoud be one of {'lcss'}
    - constarint (str, optional): the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    - w (float, optional): If :code:`constraint = "Sakoe-Chiba"` , :code:`w` is the largest temporal shift allowed between two time series; if  :code:`constraint = "Itakura"`, :code:`w` is the slope of the "Itakura Parallelogram", default to 10000.
    - epsilon (float, required if :code:`constraint = "lcss"`, otherwise not needed): the matching threshold for Longest Common Subsequence (LCSS) measure
    
This function fit the 1NN classifier from the training dataset

:param X: training dataset
:type X: np.array
:param Xlabel: target values (labels)
:type Xlabel: np.array
:return: Fitted 1NN classifier

Predic class lables for given dataset

:param X: test samples
:type X: np.array
:return: Predicted class labels for each data sample


Edit Distance with Real Penalty (ERP) [1]_ is another edit distance measure that aims to take the advantages of both DTW and other edit distance measures. 
It does this by using Lp-norm distance metrics when comparing two elements or comparing each element to a gap variable, m. 
ERP is a metric, meaning that lower bounding is possible through the triangle inequality. 
This is very useful for pruning through clustering and classfication algorithms.

ERP provides an advantage over other edit distance measures by providing exact differences between values. 
Additionally, ERP has no :math:`\epsilon` value to tune. 
Instead, one has to set a gap variable which is often set to 0 to provide intuitive results.

Lastly, ERP is also very editable; 
in the formula below, Euclidean distance is used as the internal distance measure but other measures such as absolute difference are also compatible with ERP. 
This would change :math:`D` and :math:`\pi` but other metrics might have more desirable properties to certain users.

.. math::

    \begin{equation*}
        D^u & = 0\\
        D^h(x,y,m) & = (x - m)^2\\
        D^v(x,y,m) & = (y - m)^2\\
        D^d(x,y,m) & = (x - y)^2\\
        \pi(d_{n,m}) & = \sqrt{d_{n,m}}
    \end{equation*}

:param X: a time series
:type X: np.array
:param Y: another time series
:type Y: np.array
:param m: the gap variable
:type m: float
:param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
:type constraint: float, optional
:param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
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

        lcss_dist = lcss(X, Y, epsilon = 0.7)
        lcss_dist

Output:

.. code-block:: bash

    34.61213659975356

**References**

.. [1] Lei  Chen  and  Raymond  Ng.  “On  The  Marriage  of  Lp-norms  and  EditDistance”. In:Proceedings of the 30th VLDB Conference,Toronto, Canada. (2004)


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

    X = np.array([3, 4, 38, 4, 5])
    Y = np.array([0, 3, 4])

    edr_dist = erp(X, Y, m = 4)
    edr_dist

Output:

.. code-block:: bash

    3.0

        

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
:param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
:type constraint: float, optional
:param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
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

..[1] Pierre-Fran ̧cois  Marteau.  “Time  Warp  Edit  Distance  with  Stiffness  Ad-justment  for  Time  Series  Matching”.  In:IEEE Transactions on PatternAnalysis and Machine Intelligence31.306 - 318 (2009)

Move-Split-Merge (MSM) [1]_ is an edit distance measure that deconstructs the popular editing operations (insert, delete, and substitute);
nstead it proposes sub-operations that have can be used in conjunctions to replicate the original operations. 
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
    where $\overline{x}$ is the new value and 
    \begin{equation*}
        Cost(split) = Cost(merge) = c
    \end{equation*}
    where ${c}$ is a set constant.

:param X: a time series
:type X: np.array
:param Y: another time series
:type Y: np.array
:param c: the cost for one *move* or *split** operation 
:type c: float
:param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
:type constraint: float, optional
:param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
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

Sequence Weighted Alignment (SWALE) [1]_ is an $\epsilon$ based distance measure. 
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
:param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
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

Weighted dynamic time warping (WDTW) [1]_ is a variation of DTW 
that aims to give more importance to the shape similarity of two time series. 
It does this through a weighted vector that penalizes the differences between i and j.

.. math::

    \begin{aligned}
        D(x_i,y_j,w_{abs(i-j)) = w_{abs(i-j) * |x_i - y_j|\\
        \pi(d_{i,j}) = d_{i,j}
    \end{aligned}

Note: w is a element of a weight vector whose elements are calculated as:

.. math::

    \begin{equation*}
        w_i(g,|X|) = \frac{1}{1 + e^{-g * (i - \frac{|X|}{2})}}
    \end{equation*}

where $|X|$ is the length of the time series X.

:param X: a time series
:type X: np.array
:param Y: another time series
:type Y: np.array
:param g: a constant that determines the weight vector. (see the formula above)
:type g: float
:return: WDTW Distance

**Example:**

Input:

.. code-block:: python

    X = np.array([3, 4, 76, 4, 5])
    Y = np.array([0, 3, 4])

    wdtw_distance = wdtw(X, Y, g = 0.25)
    print(wdtw_distance)

Output:

.. code-block:: bash

    1.0459354060018373

**References:**

.. [1] Young-Seon Jeong, Myong K. Jeong, and Olufemi A. Omitaomu. “Weighteddynamic time warping for time series classification”. In:Pattern Recognition4.2231 – 2240 (2011)