
#################
Sliding Measures
#################

Sliding measures [1]_ define the distance between time series :math:`X` and time series :math:`Y` 
by finding the largest correlation between :math:`X` and all shifted versions of :math:`Y` , 
where each shifted version is created by moving all entries in :math:`Y`  towards right by :math:`s` positions. 
In this process, we create a cross-correlation sequence, :math:`CC_{w}(\vec{x}, \vec{y})` with :math:`w\in{1, 2, ..., 2m-1}` 
of length :math:`(2m-1)` that contains the inner product of two time series in every possible shift. 

Normalized Cross-Correlation
====================================

.. automodule:: tsd_methods.sliding

    .. autofunction:: NCC


Biased Normalized Cross-Correlation
====================================

.. automodule:: tsd_methods.sliding

    .. autofunction:: NCCb


Unbiased Normalized Cross-Correlation
======================================

.. automodule:: tsd_methods.sliding

    .. autofunction:: NCCu


Coefficient Normalized Cross-Correlation
=========================================

.. automodule:: tsd_methods.sliding

    .. autofunction:: NCCc


**Reference**
.. [1] John Paparrizos et al. “Debunking Four Long-Standing Misconceptions ofTime-Series Distance Measures”. In:ACM SIGMOD(2020)