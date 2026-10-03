import numpy as np
import pytest

from linalg_utils import skew


def random_vectors(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=3) for _ in range(n)]


def test_skew_known():
    np.testing.assert_allclose(
        skew(np.array([1, 2, 3])),
        [[0, -3, 2], [3, 0, -1], [-2, 1, 0]],
    )


@pytest.mark.parametrize("a", random_vectors())
def test_skew_antisymmetric(a):
    np.testing.assert_allclose(skew(a).T, -skew(a))


@pytest.mark.parametrize("a", random_vectors())
def test_skew_is_cross_product(a):
    b = np.array([0.4, -1.5, 2.2])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-12)


@pytest.mark.parametrize("a", random_vectors())
def test_skew_self_cross_is_zero(a):
    np.testing.assert_allclose(skew(a) @ a, np.zeros(3), atol=1e-12)
