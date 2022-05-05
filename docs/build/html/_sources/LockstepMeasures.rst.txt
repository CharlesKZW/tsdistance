##################
Lockstep Measures
##################

Lockstep measures involve some element-wise comparison between two time series. 
This restricts the time series to those of the same length. 
Despite this restriction, lock-step measures are very versatile due to the sheer variety of them. 
Here we will provide a equation and short description of each one in our library.

The usage of Lockstep measures is consistent across all measures. 
Below is an example of using ``manhattan`` distance:

.. code-block:: python

    from tsdistance.lockstep import manhattan
    import numpy as np

    ts1 = np.array([1, 2, 3, 4, 5, 9, 7])
    ts2 = np.array([8, 9, 9, 7, 3, 1, 2])

    dist_manhattan = manhattan(ts1, ts2)
    print(dist_manhattan)

Output:

.. code-block:: bash

    38


Minkowski Functions
====================

The Minkowski functions include Euclidean distance, Manhattan Distance and Chebyshev's distance. 
They are all variations of:

.. math::

    \begin{equation*}
        (\sum_{i=1}^n |X_i - Y_i|^p)^{\frac{1}{p}}
    \end{equation*}

.. automodule:: lockstep

    .. autofunction:: minkowski

    .. autofunction:: abs_euclidean

    .. autofunction:: manhattan

    .. autofunction:: chebyshev


L1 Functions
=============
The :math:`L_1` functions all involve using the Manhattan metric in some fashion, see formula for each approach below for details. 

.. automodule:: lockstep

    .. autofunction:: sorensen

    .. autofunction:: gower

    .. autofunction:: soergel

    .. autofunction:: Kulczynski

    .. autofunction:: canberra

    .. autofunction:: lorentzian

    .. autofunction:: Intersection

Intersection Functions
=======================
The intersection family of functions have a strong relationship with the :math:`L_1` family of functions. 
Many of the intersection functions can be converted to :math:`L_1` by replacing :math:`\min(X_i,Y_i)` with :math:`\frac{|X_i, Y_i|}{2}`. 
One commonality between the intersection family of functions is the use of the element-wise minimum of the two time series. 

.. automodule:: lockstep

    .. autofunction:: wave_hedges

    .. autofunction:: czekanowski

    .. autofunction:: motyka

    .. autofunction:: tanimoto

Inner Product Functions
========================
The inner product functions all use the sum of pairwise multiplication of the elements from both time series.

.. automodule:: lockstep

    .. autofunction:: innerproduct

    .. autofunction:: harmonicmean

    .. autofunction:: kumarhassebrook

    .. autofunction:: jaccard

    .. autofunction:: cosine

    .. autofunction:: dice

Squared Chord Functions
=======================
The Squared Chord functions are a set of geometric mean distances. Thus, these distance functions are not compatible with negative values in either time series.

.. automodule:: lockstep

    .. autofunction:: fidelity

    .. autofunction:: bhattacharyya

    .. autofunction:: Square_chord

    .. autofunction:: hellinger

    .. autofunction:: matusita

Squared L2 Functions
=====================
The squared :math:`L_2` distance functions are a group of distance measures that all have :math:`(X_i - Y_i)^2` as the base.

.. automodule:: lockstep

    .. autofunction:: squared_euclidean

    .. autofunction:: clark

    .. autofunction:: neyman

    .. autofunction:: pearson

    .. autofunction:: squared_chi

    .. autofunction:: K_divergence

    .. autofunction:: additive_symm_chi

    .. autofunction:: prob_symmetric_chi

Shannon's Enthropy Functions
=============================
The following functions are based on Shannon's Entropy metric which has to deal with how much information a variable contains and the probabilistic uncertainty of information.

.. automodule:: lockstep

    .. autofunction:: kullback

    .. autofunction:: jeffrey

    .. autofunction:: K_divergence

    .. autofunction:: topsoe

    .. autofunction:: jensen_shannon

    .. autofunction:: jensen_difference

Vicissitude Functions
=====================
This group of functions is based on Vicis-Wave Hedges function.

.. automodule:: lockstep

    .. autofunction:: vicis_wave_hedges

    .. autofunction:: emanon2

    .. autofunction:: emanon3

    .. autofunction:: emanon4

    .. autofunction:: max_symmetric_chi

    .. autofunction:: min_symmetric_chi 

Combination Functions
======================
The combination functions take approaches from multiple types of functions displayed already.

.. automodule:: lockstep

    .. autofunction:: taneja
    
    .. autofunction:: kumar_johnson

    .. autofunction:: avg_l1_linf

    
