<!-- Our title -->
<div align="center">
  <h1><i>tsdistance</i>: time series distance measures</h1>
</div>

<!-- Short description -->
<p align="center">
  tsdistance is a open-source library for time series distance measures, which can be used in various time series machine learning tasks (classification, clustering, motif discovery, etc.)
</p>

## Installation

from PyPi: ``python -m pip install tsdistance``

## Getting started

### 1. Getting the data in the right format
tsdistance expects a time series to be formatted as a 1D `numpy` array. Distance measures in `tsdistance.elastic` can take in time series of different length as input, while distance measures in other sections expect input time series to have equal length. 

```python3
>>> from tsdistance.elastic import lcss
>>> import numpy as np
>>> X = np.array([3, 4, 38, 4, 5])
>>> Y = np.array([0, 3, 4])
>>> lcss_dist = lcss(X, Y, epsilon = 0.7)
>>> lcss_dist

>>> 0.33333333333333337
```


### 2. Training a model

After getting the data in the right format, a model can be trained. `tsdistance` has a built-in 1NN Classifier for experimentation purposes. For more other use cases, all distance measure functions implemented can be used as customized cost function for models in `scikit-learn` library. The following code uses  ``lcss`` to classify time series in the ``Coffee`` dataset from the UCR archive.

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
*in progress*

## Documentation

The documentation is hosted at [[tsdistance]](https://flourishing-ganache-f26bde.netlify.app/index.html). It includes an API and a user guide.


