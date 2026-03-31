import numpy as np

from tsdistance import OneNN


def test_onenn_supports_string_metrics_and_preserves_training_order():
    x0 = np.array([0.0, 1.0, 2.0])
    x1 = np.array([10.0, 11.0, 12.0])
    train = [x0, x1]
    labels = np.array([0, 1])

    clf = OneNN("dtw")
    clf.fit(train, labels)
    before = clf.Xlabel.copy()
    prediction = clf.predict([np.array([0.0, 1.0, 2.1])])

    np.testing.assert_array_equal(clf.Xlabel, before)
    np.testing.assert_array_equal(prediction, np.array([0]))


def test_onenn_reports_pruning_when_lower_bound_is_used():
    train = [
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([10.0, 11.0, 12.0, 13.0]),
        np.array([20.0, 21.0, 22.0, 23.0]),
    ]
    labels = np.array([0, 1, 2])
    query = [np.array([0.0, 1.1, 2.0, 2.9])]

    clf = OneNN("dtw", metric_param=(None, False), lb_metric="lb_kim")
    clf.fit(train, labels)
    prediction, pruning_power = clf.predict(query)

    np.testing.assert_array_equal(prediction, np.array([0]))
    assert 0.0 <= pruning_power <= 1.0
