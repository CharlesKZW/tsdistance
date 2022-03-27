.. _LowerBounds:

########################
Lower Bounding Measures
########################

Time series classification tasks usually rely on Nearest Neighbors Search, 
which classifies query series by computing their distances to all series in the training set 
and label query series using the class label of their nearest training sample. 
This process is sometimes time-consuming 
because elastic measures used for each comparison is in :math:`O(L^2)` 
where :math:`L` is the length of a single time series. 
To speed up Nearest Neighbors Search, 
Lower Bounding Measures first compute a *lower bound* of the similarity between two series 
with :math:`O(L)` complexity and go on to compute the actual :math:`O(L^2)`  similarity only when 
the *lower bound* is less than the distance between the query series and its "nearest neighbor" in the training set.

One Nearest Neighbor Classifier with Lower Bounds
=======================================================
.. automodule:: LowerBounds

    .. autoclass:: Bounded1NN
       :members: fit, predict


**Example**

Input

.. code-block:: python

    # "Coffee" is one of the UCR Archive datasets.

    >>> model = Bounded1NN(metric = 'lcss')
    >>> model.fit(Coffee_train_X, Coffee_train_y)
    >>> predicted_label = model.predict(Coffee_test_X) 
    >>> print('predicted_label: ', predicted_label)

Output:

.. code-block:: bash

    lb_predict:  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

:code:`sklearn.neighbors.KNeighborsClassifier` in the :code:`Scikit-Learn` library is a popular tool for Nearest Neighbor Search, :code:`Bounded1NN` gives the same accuracy and is about 5 times faster when using :code:`lcss` for classification tasks.

Lower Bounding Longest Common Subsequence (LCSS) 
=================================================

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

Lower Bounding Edit Distance with Real Penalty (ERP)
====================================================

The algorithm for computing the Lower Bounding Edit Distance with Real Penalty (ERP) is:

.. code-block:: python

    def ERPLowerBound(x, y):

        return abs(np.sum(x) - np.sum(y))