from pprint import pprint

import numpy as np

from src.ao_admm import CPDADMM, CPDAOADMM
from src.cpdecomposition import FactorMatrix


def main():
    """
    Main function
    """

    # Natural inputs
    data_tensor = np.random.randn(3, 5, 4)
    tensor_rank = 2
    constraints = (
        "nonnegative-l1sparsity-aoadmmasc",
        "nonnegative",
        "nonnegative")
    hyperparams = [
        {"l1_lambda": 0.5},
        {},
        {},
    ]
    tolerance_error = (1e-2, 1e-2, 1e-2)
    n_iters_admm = (100, 0, 0)

    tensor_shape = data_tensor.shape
    tensor_order = data_tensor.ndim
    tensor_norm = np.linalg.norm(data_tensor)
    tensor_mean = np.mean(data_tensor)

    # Create the ADMM objects
    # At each mode, the admms object requires the following:
    #   - tensor_mode: the mode associated with the admms object
    #   - constraint: the constraint at said mode
    #   - tol_error: the tolerance error for end of the loop
    #   - n_iters: the number of iterations for ADMM convergence
    admm_list = [
        CPDADMM(
            tensor_mode=mode,
            constraint=constraints[mode],
            hyperparams=hyperparams[mode],
            tol_error=tolerance_error[mode],
            n_iters=n_iters_admm[mode],
        )
        for mode in range(tensor_order)
    ]

    # Create an CPDAOADMM object, this requires:
    #   - Input tensor
    #   - Tensor rank
    #   - List of ADMM objects, each containing:
    #       - tensor_mode: the mode associated with the admms object
    #       - constraint: the constraint at said mode
    #       - tol_error: the tolerance error for end of the loop
    #       - n_iters: the number of iterations for ADMM convergence
    #   - Number of iterations
    n_iters_ao = 100
    ao_admm = CPDAOADMM(
        tensor=data_tensor,
        tensor_rank=tensor_rank,
        admms=admm_list,
        n_iters=n_iters_ao,
    )
    ao_admm()


if __name__ == "__main__":
    main()
    print("Done")
    exit(0)
