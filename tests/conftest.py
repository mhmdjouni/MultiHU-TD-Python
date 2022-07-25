import numpy as np
import tensorly as tl
import pytest
from sklearn.preprocessing import normalize

from src.admm import ADMM
from src.ao_admm import AOADMMASC, AOADMMASCNaive


@pytest.fixture()
def load_tensor_3() -> np.ndarray[tuple[int, int, int], np.float64]:
    """
    Load factor matrices
    """
    return np.arange(60).astype("float64").reshape((3, 5, 4,))


@pytest.fixture()
def load_factor_matrices_3(
    load_tensor_3,
) -> list[
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
        np.arange(shape[0] * rank)
        .astype("float64")
        .reshape((shape[0], rank,)),
        np.arange(shape[1] * rank)
        .astype("float64")
        .reshape((shape[1], rank,)),
        np.arange(shape[2] * rank)
        .astype("float64")
        .reshape((shape[2], rank,)),
    ]


@pytest.fixture()
def load_aoadmm() -> tuple[
    list[
        np.ndarray[tuple[int, int], np.float],
        np.ndarray[tuple[int, int], np.float],
        np.ndarray[tuple[int, int], np.float],
    ],
    np.ndarray[tuple[int, int, int], np.float],
    list[ADMM, ADMM, ADMM],
    AOADMMASC,
    AOADMMASCNaive,
]:
    """
    Load ADMM and AOADMM
    """
    mode = 3
    dims = (3, 10, 5)
    rank = 2

    factors = [
        np.arange(dims[i] * rank).reshape((dims[i], rank)) for i in range(mode)
    ]
    factors[-2] = normalize(factors[-2], norm="l1", axis=1)

    weights, factors = tl.cp_tensor.cp_normalize((None, factors))

    tensor = tl.cp_tensor.cp_to_tensor((weights, factors))

    constraints = (
        "nonnegative",
        "nonnegative-l1sparsity-aoadmmasc",
        "nonnegative",
    )
    hyperparams = [
        {},
        {"l1_lambda": 0.5},
        {},
    ]
    tolerance_error = (1e-2, 1e-2, 1e-2)
    n_iters_admm = (np.inf, 100, np.inf)
    n_iters_ao = 100

    admm_list = [
        ADMM(
            tensor_mode=mode,
            constraint=constraints[mode],
            hyperparams=hyperparams[mode],
            tol_error=tolerance_error[mode],
            n_iters=n_iters_admm[mode],
        )
        for mode in range(tensor.ndim)
    ]

    ao_admm_asc = AOADMMASC(
        tensor=tensor, tensor_rank=rank, admms=admm_list, n_iters=n_iters_ao,
    )

    ao_admm_ascnaive = AOADMMASCNaive(
        tensor=tensor, tensor_rank=rank, admms=admm_list, n_iters=n_iters_ao,
    )

    return (
        factors,
        tensor,
        admm_list,
        ao_admm_asc,
        ao_admm_ascnaive,
    )
