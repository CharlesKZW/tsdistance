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
        self.metric = metric
        self.lb_metric = lb_metric
        self.metric_param = metric_param
        self.lb_param = lb_param

    def fit(self, X, Xlabel):
        r"""
        This function fits the 1NN classifier from the training dataset.

        :param X: training dataset
        :type X: np.array
        :param Xlabel: target values (labels)
        :type Xlabel: np.array
        :return: Fitted 1NN classifier
        """
        self.X = X
        self.Xlabel = Xlabel

    def predict(self, Y):
        r"""
        Predic class lables for given dataset

        :param X: test samples
        :type X: np.array
        :return: Predicted class label for each data sample 

        """

        if self.lb == True:
            pruned = 0

        test_class = np.zeros(Y.shape[0])

        for idx_y, y in enumerate(Y):

            best_so_far = float("inf")

            lb_list = np.zeros(self.X.shape[0])

            for idx_x, x in enumerate(self.X):

                lb_dist = self.lb_metric(x, y, *self.lb_param)

                lb_list[idx_x] = lb_dist

            ordering = np.argsort(lb_list)

            self.X = self.X[ordering]
            self.Xlabel = self.Xlabel[ordering]
            lb_list = lb_list[ordering]

            for idx_x, x in enumerate(self.X):

                lb_dist = lb_list[idx_x]

                if lb_dist < best_so_far:

                    actual_dist = self.metric(x, y, *self.metric_param)

                    if actual_dist < best_so_far:
                        best_so_far = actual_dist
                        test_class[idx_y] = self.Xlabel[idx_x]

                if self.lb == True and lb_dist > best_so_far:
                    pruned += 1

                pruning_power = pruned / (Y.shape[0] * self.X.shape[0])

                if self.lb == False:

                    actual_dist = self.lb_metric(x, y, *self.lb_param)
                    if actual_dist < best_so_far:
                        best_so_far = actual_dist
                        test_class[idx_y] = self.Xlabel[idx_x]

                    pruning_power = 0

        if self.lb == True:
            return test_class, pruning_power
        if self.lb == False:
            return test_class
