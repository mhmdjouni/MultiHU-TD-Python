from pprint import pprint

import numpy as np
import numpy.linalg as la
import tensorly.cp_tensor as tlcp
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from src.admm import ADMM
from src.ao_admm import AOADMMASC, AOADMMASCNaive, AOADMM


def main_aoadmm():
    """
    Main function
    """

    mode = 3
    dims = (4, 4, 4)
    rank = 2

    factors_orig = [
        np.abs(np.random.randn(dims[i], rank)) for i in range(mode)
    ]

    core_orig, factors_orig = tlcp.cp_normalize((None, factors_orig))
    core_indices = np.argsort(core_orig)[::-1]
    core_orig = core_orig[core_indices]
    factors_orig = [fm[:, core_indices] for fm in factors_orig]

    tensor_orig = tlcp.cp_to_tensor((core_orig, factors_orig))

    constraints = (
        "nonnegative",
        "nonnegative",
        "nonnegative",
    )
    hyperparams = [
        {},
        {"l1_lambda": 0.5},
        {},
    ]
    tolerance_error = (1e-3, 1e-3, 1e-3)
    n_iters_admm = (np.inf, np.inf, np.inf)
    n_iters_ao = 500

    admm_list = [
        ADMM(
            tensor_mode=mode,
            constraint=constraints[mode],
            hyperparams=hyperparams[mode],
            tol_error=tolerance_error[mode],
            n_iters=n_iters_admm[mode],
        )
        for mode in range(tensor_orig.ndim)
    ]

    ao_admm = AOADMM(
        tensor=tensor_orig,
        tensor_rank=rank,
        admms=admm_list,
        n_iters=n_iters_ao,
    )

    ao_admm.solve()

    print("")

    print("Matrix A")
    print(factors_orig[1])
    print(ao_admm.factors[1])
    print(f"Norm: {la.norm(ao_admm.factors[1], axis=0)}")
    rec_error = la.norm(factors_orig[1] - ao_admm.factors[1]) / la.norm(
        factors_orig[1]
    )
    print(f"Reconstruction error: {rec_error}")

    print("")

    print("Matrix B")
    print(factors_orig[2])
    print(ao_admm.factors[2])
    print(f"Norm: {la.norm(ao_admm.factors[2], axis=0)}")
    rec_error = la.norm(factors_orig[2] - ao_admm.factors[2]) / la.norm(
        factors_orig[2]
    )
    print(f"Reconstruction error: {rec_error}")

    print("")

    print("Matrix C")
    print(factors_orig[0])
    print(ao_admm.factors[0])
    print(f"Norm: {la.norm(ao_admm.factors[0], axis=0)}")
    rec_error = la.norm(factors_orig[0] - ao_admm.factors[0]) / la.norm(
        factors_orig[0]
    )
    print(f"Reconstruction error: {rec_error}")

    print("")

    print(f"Reconstruction Error: {ao_admm.recons_error[-1]}")

    print("")

    plt.figure()
    plt.plot(ao_admm.recons_error)
    plt.show()

    pass


def main_aoadmmasc():
    """
    Main function
    """

    mode = 3
    dims = (4, 4, 4)
    rank = 2

    factors_orig = [
        np.abs(np.random.randn(dims[i], rank)) for i in range(mode)
    ]
    factors_orig[1] = normalize(factors_orig[1], axis=1, norm="l1")

    core_orig, [factors_orig[0], factors_orig[2]] = tlcp.cp_normalize(
        (None, [factors_orig[0], factors_orig[2]])
    )
    core_indices = np.argsort(core_orig)[::-1]
    core_orig = core_orig[core_indices]
    factors_orig = [fm[:, core_indices] for fm in factors_orig]

    tensor_orig = tlcp.cp_to_tensor((core_orig, factors_orig))

    constraints = (
        "nonnegative",
        "nonnegative-aoadmmasc",
        "nonnegative",
    )
    hyperparams = [
        {},
        {"l1_lambda": 0.5},
        {},
    ]
    tolerance_error = (1e-3, 1e-3, 1e-3)
    n_iters_admm = (np.inf, np.inf, np.inf)
    n_iters_ao = 500

    admm_list = [
        ADMM(
            tensor_mode=mode,
            constraint=constraints[mode],
            hyperparams=hyperparams[mode],
            tol_error=tolerance_error[mode],
            n_iters=n_iters_admm[mode],
        )
        for mode in range(tensor_orig.ndim)
    ]

    ao_admm_asc = AOADMMASC(
        tensor=tensor_orig,
        tensor_rank=rank,
        admms=admm_list,
        n_iters=n_iters_ao,
    )

    ao_admm_asc.solve()

    print("")

    print("Matrix A")
    print(factors_orig[1])
    print(ao_admm_asc.factors[1])
    print(f"Norm: {la.norm(ao_admm_asc.factors[1],ord=1, axis=1)}")
    rec_error = la.norm(factors_orig[1] - ao_admm_asc.factors[1]) / la.norm(
        factors_orig[1]
    )
    print(f"Reconstruction error: {rec_error}")

    print("")

    print("Matrix B")
    print(factors_orig[2])
    print(ao_admm_asc.factors[2])
    print(f"Norm: {la.norm(ao_admm_asc.factors[2], axis=0)}")
    rec_error = la.norm(factors_orig[2] - ao_admm_asc.factors[2]) / la.norm(
        factors_orig[2]
    )
    print(f"Reconstruction error: {rec_error}")

    print("")

    print("Matrix C")
    print(factors_orig[0])
    print(ao_admm_asc.factors[0])
    print(f"Norm: {la.norm(ao_admm_asc.factors[0], axis=0)}")
    rec_error = la.norm(factors_orig[0] - ao_admm_asc.factors[0]) / la.norm(
        factors_orig[0]
    )
    print(f"Reconstruction error: {rec_error}")

    print("")

    print(f"Reconstruction Error: {ao_admm_asc.recons_error[-1]}")

    print("")

    plt.figure()
    plt.plot(ao_admm_asc.recons_error)
    plt.show()

    pass


if __name__ == "__main__":
    # main_aoadmm()
    # print("Done with AOADMM")
    main_aoadmmasc()
    print("Done with AOADMMASC")
    exit(0)
