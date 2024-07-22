<!-- Our title -->
<div align="center">
  <h1><i>tsdistance</i>: time series distance measures</h1>
</div>

<!-- Short description -->
  tsdistance provides distance measures (a.k.a. metrics) suitable for various time series machine learning tasks, including classification, clustering, motif discovery, similarity search, etc. 
  
  For details, please see the [documentation](https://tsdistance.readthedocs.io/en/latest/).


## Installation

from PyPi: ``python -m pip install tsdistance``

## Examples

### 1. Compute the distance between two time series


```python3
>>> from tsdistance.elastic import lcss
>>> import numpy as np
>>> X = np.array([3, 4, 38, 4, 5])
>>> Y = np.array([0, 3, 4])
>>> lcss_dist = lcss(X, Y, epsilon = 0.7)
>>> lcss_dist

>>> 0.33333333333333337
```


### 2. Use a distance measure in a machine learning model (e.g. classfication)

```python3
>>> from tsdistance import OneNN
>>> model = OneNN(metric = 'lcss')
>>> model.fit(Coffee_train_X, Coffee_train_y)
>>> predicted_label = model.predict(Coffee_test_X)
>>> print('predicted_label: ', predicted_label)

>>> lb_predict:  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```


## Available features

**Elastic Measures**

- Dynamic Time Warping (DTW);
- Longest Common Subsequence (LCSS);
- Edit Distance with Real Penalty (ERP);
- Edit Distance on Real Sequences (EDR);
- Time Warp Edit Distance (TWED);
- Move-Split-Merge (MSM);
- Sequence Weighted Alignment (SWALE);
- Weighted Dynamic Time Warping (WDTW)

**Lockstep Measures**
- Minkowski Functions;
- L1 Functions;
- Intersection Functions;
- Inner Product Functions;
- Squared Chord Functions;
- Squared L2 Functions;
- Shannon’s Enthropy Functions;
- Vicissitude Functions;
- Combination Functions;

**Sliding Measures**
- Normalized Cross-Correlation;
- Biased Normalized Cross-Correlation;
- Unbiased Normalized Cross-Correlation;
- Coefficient Normalized Cross-Correlation;

**Kernel Measures**
- Kernel Dynamic Time Warping (kdtw);
- Shift INvariant Kernel (SINK);
- Log Global Alignment Kernel (LGAK)

**Multivariate Distance Measures**
- *in progress*





