import math

import numpy as np

from tsdistance import kernel


# Sources:
# - SINK / GRAIL: https://www.vldb.org/pvldb/vol12/p1762-paparrizos.pdf
# - GAK / tslearn reference source:
#   https://raw.githubusercontent.com/tslearn-team/tslearn/65ac4baa44b082e6a5832372f9a8bbe3f448b707/tslearn/metrics/softdtw_variants.py


def preserved_energy_reference(x, e):
    arr = np.asarray(x, dtype=float)
    fft = np.fft.fft(arr, 2 ** kernel.nextpow2(2 * len(arr) - 1))
    power = np.abs(fft) ** 2
    cutoff = int(np.argwhere(np.cumsum(power) / np.sum(power) >= (e / 2.0))[0][0])
    filtered = fft.copy()
    filtered[cutoff + 1 : len(filtered) - cutoff - 1] = 0
    return filtered


def sink_reference(x, y, gamma, e):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fx = preserved_energy_reference(x_arr, e)
    fy = preserved_energy_reference(y_arr, e)

    def sum_ncc(a_fft, a_raw, b_fft, b_raw):
        corr = np.fft.ifft(a_fft * np.conj(b_fft)).real / (
            np.linalg.norm(a_raw) * np.linalg.norm(b_raw)
        )
        return float(np.sum(np.exp(gamma * corr)))

    return sum_ncc(fx, x_arr, fy, y_arr) / math.sqrt(
        sum_ncc(fx, x_arr, fx, x_arr) * sum_ncc(fy, y_arr, fy, y_arr)
    )


def lgak_reference(x, y, sigma):
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
    squared = (x_arr - y_arr.T) ** 2
    gram = -squared / (2 * sigma**2)
    gram -= np.log(2 - np.exp(gram))
    gram = np.exp(gram)
    csum = np.zeros((gram.shape[0] + 1, gram.shape[1] + 1), dtype=float)
    csum[0, 0] = 1.0
    for i in range(gram.shape[0]):
        for j in range(gram.shape[1]):
            csum[i + 1, j + 1] = (
                csum[i, j + 1] + csum[i + 1, j] + csum[i, j]
            ) * gram[i, j]
    return float(csum[-1, -1])


def test_kdtw_distance_uses_documented_sigma_scaling():
    expected = (1.0 / 3.0) * math.exp(-((2.0 - 1.0) ** 2) / 2.0)
    assert math.isclose(kernel.kdtw_distance(2.0, 1.0, 2.0), expected, rel_tol=1e-12)


def test_sink_matches_direct_definition_and_is_normalized():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 2.0, 4.0])
    expected = sink_reference(x, y, gamma=0.5, e=2.0)
    assert math.isclose(kernel.SINK(x, y, gamma=0.5, e=2.0), expected, rel_tol=1e-12)
    assert math.isclose(kernel.SINK(x, x, gamma=0.5, e=2.0), 1.0, rel_tol=1e-12)


def test_lgak_matches_tslearn_gram_recurrence():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 2.0, 3.0])
    expected = lgak_reference(x, y, sigma=2.0)
    assert math.isclose(kernel.LGAK(x, y, sigma=2.0), expected, rel_tol=1e-12)
