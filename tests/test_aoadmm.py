import numpy as np

from src.admm import ADMM
from src.ao_admm import AOADMMASC, AOADMMASCNaive


def test_load_aoadmm(load_aoadmm):
    """
    Test load_aoadmm
    """
    (factors, tensor, admm_list, ao_admm_asc, ao_admm_ascnaive,) = load_aoadmm

    assert factors[-2].shape == (10, 2)
    assert factors[-1].shape == (5, 2)
    assert factors[-3].shape == (3, 2)
    assert tensor.shape == (3, 10, 5)
    assert isinstance(admm_list[-3], ADMM)
    assert isinstance(ao_admm_asc, AOADMMASC)
    assert isinstance(ao_admm_ascnaive, AOADMMASCNaive)


def test_aoadmmasc(load_aoadmm):

    (factors_orig, tensor_orig, _, ao_admm_asc, _,) = load_aoadmm

    dims = list(tensor_orig.shape)
    rank = factors_orig[-2].shape[1]

    factors, tensor_unfoldings, dual_vars = ao_admm_asc._initialize_solver()
    dims[-1] += 1

    for mode in range(3):
        assert factors[mode].shape == (dims[mode], rank)
        assert tensor_unfoldings[mode].shape == (
            np.prod(dims[:mode] + dims[mode + 1 :]),
            dims[mode],
        )
        assert dual_vars[mode].shape == (dims[mode], rank)

    assert np.allclose(np.linalg.norm(factors[-2], ord=1, axis=1), 1)
    assert np.allclose(np.linalg.norm(factors[-1][:-1, :], axis=0), 1)
    assert np.allclose(np.linalg.norm(factors[-3], axis=0), 1)


def test_aoadmm(load_aoadmm):

    (factors, tensor, admm_list, ao_admm_asc, ao_admm_ascnaive,) = load_aoadmm

    print("")

    ao_admm_asc.solve()

    assert ao_admm_asc.factors[-2].shape == (10, 2)
    assert ao_admm_asc.factors[-1].shape == (5, 2)
    assert ao_admm_asc.factors[-3].shape == (3, 2)

    print(ao_admm_asc.recons_error)
    # assert ao_admm_asc.recons_error[-1] < 1e-1

    print("")

    ao_admm_ascnaive.solve()

    assert ao_admm_ascnaive.factors[-2].shape == (10, 2)
    assert ao_admm_ascnaive.factors[-1].shape == (5, 2)
    assert ao_admm_ascnaive.factors[-3].shape == (3, 2)

    print(ao_admm_ascnaive.recons_error)
    # assert ao_admm_ascnaive.recons_error[-1] < 1e-6
