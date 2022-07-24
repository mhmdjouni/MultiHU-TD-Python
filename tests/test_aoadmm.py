from src.admm import ADMM
from src.ao_admm import AOADMMASC, AOADMMASCNaive


def test_load_aoadmm(load_aoadmm):
    """
    Test load_aoadmm
    """
    (
        factors,
        tensor,
        admm_list,
        ao_admm_asc,
        ao_admm_ascnaive,
    ) = load_aoadmm

    assert factors[0].shape == (10, 2)
    assert factors[1].shape == (5, 2)
    assert factors[2].shape == (3, 2)
    assert tensor.shape == (10, 5, 3)
    assert isinstance(admm_list[0], ADMM)
    assert isinstance(ao_admm_asc, AOADMMASC)
    assert isinstance(ao_admm_ascnaive, AOADMMASCNaive)
