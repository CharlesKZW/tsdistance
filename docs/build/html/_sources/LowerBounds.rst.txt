.. _LowerBounds:

######################################
One Nearest-Neighbor Classifier
######################################

Time series classification tasks usually rely on Nearest Neighbors Search, 
which classifies query series by computing their distances to all series in the training set 
and label query series using the class label of their nearest training sample. 
Here we provide a  

One Nearest Neighbor Classifier
=================================

.. automodule:: OneNN

    .. autoclass:: OneNN
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
