import numpy as np
import pytest


@pytest.fixture()
def load_tensor_3() -> np.ndarray[tuple[int, int, int], np.float64]:
    """
    Load factor matrices
    """
    return np.arange(60).astype('float64').reshape((3, 5, 4,))


@pytest.fixture()
def load_factor_matrices_3(load_tensor_3) -> list[
    np.ndarray[tuple[int, int], np.float64],
    np.ndarray[tuple[int, int], np.float64],
    np.ndarray[tuple[int, int], np.float64],
]:
    """
    Load factor matrices
    """
    my_tensor = load_tensor_3
    shape = my_tensor.shape
    rank = 4

    return [
        np.arange(shape[0]*rank).astype('float64').reshape((shape[0], rank,)),
        np.arange(shape[1]*rank).astype('float64').reshape((shape[1], rank,)),
        np.arange(shape[2]*rank).astype('float64').reshape((shape[2], rank,)),
    ]
