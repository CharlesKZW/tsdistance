.. _elastic_measures:

################
Elastic Measures
################


Elastic measures are popular for time series analysis.
Using elastic measures, the two time series do not need to be the same length, providing much more flexibility.
The elastic measures implemented in this library include Dynamic Time Warping (DTW),
its variations such as weighted-DTW and derivative-DTW,
and edit-based distance measures such as Longest Common Subsequence (LCSS),
Edit Distance with Real Penalty (ERP), Edit Distance on Real Sequences (EDR), Move Split Merge (MSM),
and Time Warp Edit Distance (TWED).

General Formula
===============
In general, elastic measures use a dynamic programming algorithm with the goal of minimize the total distance between the two series.
This has an effect of "aligning" the time series which is useful for time-series that are out-of-sync
or otherwise unaligned in ways that non-elastic measures could not distinguish.

Through this section, we will have two time series X and Y represented as such:

.. math::

    \begin{aligned}
        X = x_1,\ x_2,\ x_3,\ ...,\ x_n \\
        Y = y_1,\ y_2,\ y_3,\ ...,\ y_m \\
    \end{aligned}


Additionally, we will have a array of size :math:`(m,n)` represented as:

.. math::
    \begin{equation*}
        DP = d_{1,1},\ d_{1,2},\ ...,\ d_{1,m},\ d_{2,m},\ ...,\ d_{n,m}
    \end{equation*}
    

As the elastic measures runs, it produces a warping path :math:`W = w_1, w_2, ..., w_k = (x_1,y_1), (x_i, y_i), ..., (x_n,y_m)`
where :math:`max(m,n) \leq k \leq m + n - 1`. In warping path element, the algorithm takes one of three steps.
If :math:`w_{k-1} = (x_{i-1}, y_i)`, then the step was horizontal. If :math:`w_{k-1} = (x_{i}, y_{i-1})`, then the step was vertical.
And if :math:`w_{k-1} = (x_{i-1}, y_{i-1})`, the step was diagonal.
Along with this, each elastic measure uses a distance function D to determine the distance between the elements of each time series.
D can be divided into 4 components, :math:`D^{v},D^{h},D^d`, and $D^u$ or distance from a vertical step, distance from a horizontal step,
distance from a diagonal step, and undirectional distance. These components are the same in some measures and differ in others.
The elastic distance of the two time series is :math:`\sum_{i = 1}^{k}D(w_i)` where :math:`D` is one of the component listed above.
Alternatively, the elastic distance can be written as:

.. math::
    \begin{equation}
        ElasticDistance(X,Y) = \pi(d_{n,m})
    \end{equation}

where

.. math::
    \begin{equation}
    d_{i,j} =
    \begin{cases}
        D^u(x_i,y_j) & \text{if $i,j = 1$}\\
        d_{i-1,j} + D^v(x_i,y_j) & \text{if $i \neq 1$ and $j = 1$} \\
        d_{i,j-1} + D^h(x_i,y_j) & \text{if $i = 1$ and $j \neq 1$} \\
        min(d_{i-1,j-1} + D^d(x_i,y_j), d_{i-1,j}\\ + D^h(x_i,y_j),
        d_{i,j-1} + D^v(x_i,y_j)) & \text{if $i, j \neq 1$, $i \leq n$, $j \leq m$}
    \end{cases}
    \end{equation}

The function :math:`\pi` is a final function depending on the measure.
Throughout this paper, we will define :math:`\pi` and D as a simplistic way to define each function.
If the the distance function differs depending on the direction that will be detailed in the measure.
Additionally, certain measures will have other parameters that must be specified. These will be discussed in where applicable.
In the worst case,
elastic measures will take :math:`O(mn)` time to complete as each item within the time series are compared to each other before the final comparison is computed.
There are various methods to improve this run-time, such as restricting the range of comparing values.
Additionally,
the space complexity in the worst case is an array of :math:`O(mn)`
but this can be reduced to a linear size complexity as only the immediately previous values are used.

Dynamic Time Warping (DTW) and DTW variants
====================================================

.. automodule:: tsdistance.elastic

    .. autofunction:: dtw
    
    .. autofunction:: ddtw


DTW Lower Bounds
=================

.. automodule:: tsdistance.elastic

    .. autofunction:: lb_kim
    
    .. autofunction:: lb_keogh
    
    .. autofunction:: lb_new

    .. autofunction:: lb_improved

Longest Common Subsequence (LCSS) and Derivative LCSS (DLCSS)
==============================================================
    
.. automodule:: tsdistance.elastic

    .. autofunction:: lcss

    .. autofunction:: dlcss

LCSS Lower Bounds
===================
.. automodule:: tsdistance.elastic

    .. autofunction:: lb_keogh_lcss
    
Sequence Weighted Alignment (SWALE)
====================================
.. automodule:: tsdistance.elastic

    .. autofunction:: swale

Edit Distance with Real Penalty (ERP)
=====================================

.. automodule:: tsdistance.elastic

    .. autofunction:: erp
    
ERP Lower Bounds
===================
.. automodule:: tsdistance.elastic

    .. autofunction:: lb_erp

    .. autofunction:: lb_kim_erp

    .. autofunction:: lb_keogh_erp

Edit Distance on Real Sequences (EDR)
=====================================

.. automodule:: tsdistance.elastic

    .. autofunction:: edr

Time Warp Edit Distance (TWED)
===============================
.. automodule:: tsdistance.elastic

    .. autofunction:: twed

TWED Lower Bound
===================
.. automodule:: tsdistance.elastic

    .. autofunction:: lb_twed


Move-Split-Merge (MSM)
===========================
.. automodule:: tsdistance.elastic

    .. autofunction:: msm

MSM Lower Bound
===================
.. automodule:: tsdistance.elastic

    .. autofunction:: lb_msm


