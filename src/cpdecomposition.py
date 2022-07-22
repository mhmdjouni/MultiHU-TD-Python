from dataclasses import dataclass, field

import numpy as np


@dataclass
class FactorMatrix:
    """
    Defines a factor matrix for a tensor decomposition.
    """

    factor_matrix: np.ndarray[tuple[int, int], np.float64]
    constraints: str


@dataclass
class CPDecomposition:
    """
    Sets the abstraction for a CP Decomposition of an N-D Tensor.
    An object of this class is instantiated by:
    1. A data tensor
    2. The tensor rank
    3. The constraints & hyperparameters for the factor matrices
        - "None": no constraint
        - "NN": nonnegativity
        - "L1": sparsity with L1 norm
        - "NNL1": nonnegativity with L1 norm
        - "NNL1ASC": nonnegativity with L1 norm and ASC
        - "NNL1ASC_naive": nonnegativity with L1 norm and naive ASC
    """

    data_tensor: np.ndarray[tuple[int, ...], np.float64]
    tensor_rank: int
    constraints: tuple[str, ...]

    core_diagonal: np.ndarray[tuple[int], np.float64] = field(init=False)
    factors: list[np.ndarray[tuple[int, int], np.float64], ...] = field(
        init=False
    )

    def __post_init__(self):
        self.tensor_order = self.data_tensor.ndim
        self.shape = self.data_tensor.shape

        assert (
            len(self.constraints) == self.tensor_order
        ), "Constraints must be of same length as number of tensor modes"

        # Instantiate the diagonal elements of the diagonal core tensor
        self.core_diagonal = np.ones(self.tensor_rank)

        # Instantiate the factor matrices
        self.factors = [
            np.linalg.norm(
                np.random.randn(self.shape[mode], self.tensor_rank), axis=0,
            )
            for mode in range(self.tensor_order)
        ]

    def __str__(self):
        return f"argmin_(A, B, C) " \
               f"||T - Lambda * A * B * C||_F^2" \
               f" + " \
               f"R(A) + R(B) + R(C)"

    # Get and set the factor matrices
    # Get and set the diagonal core tensor
    # Get the reconstruction error
