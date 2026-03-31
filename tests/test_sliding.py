import numpy as np

from tsdistance import sliding


def test_sliding_ncc_variants_match_numpy_reference_sequences():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 1.0, 0.0])
    corr = np.correlate(x, y, mode="full")
    overlap = np.correlate(np.ones(len(x)), np.ones(len(y)), mode="full")

    np.testing.assert_allclose(sliding.NCC(x, y), corr)
    np.testing.assert_allclose(sliding.NCCb(x, y), corr / max(len(x), len(y)))
    np.testing.assert_allclose(sliding.NCCc(x, y), corr / (np.linalg.norm(x) * np.linalg.norm(y)))
    np.testing.assert_allclose(sliding.NCCu(x, y), corr / overlap)
