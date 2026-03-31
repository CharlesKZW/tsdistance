import importlib

import numpy as np


class OneNN(object):

    r"""
    Class for implementing One Nearest Neighbors Search with Lower Bounding Measures

    :param metric: distance measure to compute similarity
    :type metric: function
    :param metric_param: parameters of distance measure (if applicable) ,default = ``None``.
    :type constraint: tuple
    :param lb_metric: lower bounding distance measure to compute similarity (only applicable if ``metric`` is one of the Elastic Measures)
    :type lb_metric: function
    :param lb_param: parameters of distance measure (if applicable) ,default = ``None``.
    :type lb_param: tuple
    
    
    """

    def __init__(self, metric, metric_param=None, lb_metric=False, lb_param=None):
        self.metric = self._resolve_metric(metric)
        self.lb_metric = self._resolve_metric(lb_metric) if lb_metric else None
        self.metric_param = self._normalize_params(metric_param)
        self.lb_param = self._normalize_params(lb_param)
        self.lb = self.lb_metric is not None

    @staticmethod
    def _normalize_params(params):
        if params is None:
            return ()
        if isinstance(params, tuple):
            return params
        if isinstance(params, list):
            return tuple(params)
        return (params,)

    @staticmethod
    def _resolve_metric(metric):
        if callable(metric):
            return metric
        if isinstance(metric, str):
            for module_name in ("elastic", "lockstep", "kernel", "sliding"):
                module = importlib.import_module(f"tsdistance.{module_name}")
                if hasattr(module, metric):
                    return getattr(module, metric)
        raise ValueError("metric must be a callable or the name of a tsdistance measure")

    def fit(self, X, Xlabel):
        r"""
        This function fits the 1NN classifier from the training dataset.

        :param X: training dataset
        :type X: np.array
        :param Xlabel: target values (labels)
        :type Xlabel: np.array
        :return: Fitted 1NN classifier
        """
        self.X = [np.asarray(sample, dtype=float) for sample in X]
        self.Xlabel = np.asarray(Xlabel)
        if len(self.X) != len(self.Xlabel):
            raise ValueError("X and Xlabel must contain the same number of samples")
        return self

    def predict(self, Y):
        r"""
        Predic class lables for given dataset

        :param X: test samples
        :type X: np.array
        :return: Predicted class label for each data sample 

        """
        queries = [np.asarray(sample, dtype=float) for sample in Y]
        predictions = np.empty(len(queries), dtype=self.Xlabel.dtype)
        pruned = 0
        total = len(queries) * len(self.X)

        for query_index, y in enumerate(queries):
            best_so_far = float("inf")
            best_label = self.Xlabel[0]

            if self.lb:
                lb_values = np.array(
                    [self.lb_metric(x, y, *self.lb_param) for x in self.X], dtype=float
                )
                ordering = np.argsort(lb_values)
                ordered_indices = ordering.tolist()
            else:
                lb_values = None
                ordered_indices = list(range(len(self.X)))

            for position, train_index in enumerate(ordered_indices):
                x = self.X[train_index]
                if self.lb:
                    lb_dist = lb_values[train_index]
                    if lb_dist > best_so_far:
                        pruned += 1
                        continue
                actual_dist = self.metric(x, y, *self.metric_param)
                if actual_dist < best_so_far:
                    best_so_far = actual_dist
                    best_label = self.Xlabel[train_index]

            predictions[query_index] = best_label

        if self.lb:
            return predictions, (pruned / total if total else 0.0)
        return predictions
