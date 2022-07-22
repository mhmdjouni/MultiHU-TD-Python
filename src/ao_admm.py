from dataclasses import dataclass, field

import numpy as np
import numpy.linalg as la
import tensorly as tl
from sklearn.preprocessing import normalize

from src.proximal_operators import proximal_update_admm


@dataclass
class CPDADMM:
    """
    Solves ADMM sub-problem for a given mode.
    This class stores the initial and current (final) states of the factor
      matrices and the dual variables for each ADMM sub-problem.
    """

    # fixed for each object
    tensor_mode: int
    constraint: str
    hyperparams: dict
    tol_error: float
    n_iters: int = -1

    # initial state of updated parameters
    factor_mat_0: np.ndarray[tuple[int, int], np.float64] = field(init=False)
    dual_var_0: np.ndarray[tuple[int, int], np.float64] = field(init=False)

    # current state of updated parameters
    factor: np.ndarray[tuple[int, int], np.float64] = field(init=False)
    dual_var: np.ndarray[tuple[int, int], np.float64] = field(init=False)

    def __call__(
        self,
        tensor_unfolding: np.ndarray[tuple[int, int], np.float64],
        kr_product: np.ndarray[tuple[int, int], np.float64],
        factor: np.ndarray[tuple[int, int], np.float64],
        dual_var: np.ndarray[tuple[int, int], np.float64],
        bsum: float = 0,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.float64],
        np.ndarray[tuple[int, int], np.float64],
    ]:
        """
        Solves the ADMM sub-problem for a given mode.
        """
        self.factor_mat_0 = factor
        self.dual_var_0 = dual_var

        rank = factor.shape[1]
        kr_hadamard = kr_product.T @ kr_product
        rho = np.trace(kr_hadamard) / rank
        L = kr_hadamard + (rho + bsum) * np.eye(rank)
        F = kr_product.T @ tensor_unfolding
        factor_conv = factor

        itr = 0
        while True:
            factor_t = (
                la.inv(L)
                @ (F + rho * (factor + dual_var).T + bsum * factor_conv.T)
            ).T

            factor_0 = factor
            factor = proximal_update_admm(
                factor=factor_t,
                dual_var=dual_var,
                rho=rho,
                constraint=self.constraint,
                hyperparams=self.hyperparams,
            )

            dual_var = dual_var + factor - factor_t

            prim_res = la.norm(factor - factor_t) / la.norm(factor)
            dual_res = la.norm(factor - factor_0) / la.norm(dual_var)

            itr += 1

            stop_criter1 = prim_res < self.tol_error
            stop_criter2 = dual_res < self.tol_error
            stop_criter3 = itr >= self.n_iters
            if (stop_criter1 and stop_criter2) or stop_criter3:
                break

        return factor, dual_var


@dataclass
class CPDAOADMM:
    """
    Solve CPD using AO (mainly through ADMM)
    Future development may include:
     - Loris-Verhoeven solver for the factor matrices
    Current version supposes the following:
     - A has nonnegativity, asc, and sparsity constraints
     - B has nonnegativity, with normalized columns (l2 norm)
     - C has nonnegativity, with normalized columns (l2 norm)
     - The norms of B and C constitute the weights of the diagonal core tensor
    """

    tensor: np.ndarray[tuple[int, ...], np.float64]
    tensor_rank: int
    admms: list[CPDADMM, ...]
    n_iters: int

    tensor_order: int = field(init=False)
    factors0: list[np.ndarray[tuple[int, int], np.float64], ...] = field(
        init=False
    )
    factors: list[np.ndarray[tuple[int, int], np.float64], ...] = field(
        init=False
    )
    diagonal: np.ndarray[tuple[int], np.float64] = field(init=False)
    recons_error: np.ndarray[tuple[int], np.float64] = field(init=False)

    def __post_init__(self):
        self.tensor_order = self.tensor.ndim
        self.tensor_mean = np.mean(self.tensor)
        self.tensor_norm = la.norm(self.tensor)

        assert len(self.admms) == self.tensor_order, (
            "Number of ADMM sub-problems must be equal to the number of "
            "tensor modes "
        )

        self.recons_error = np.zeros(self.n_iters)

    def __initialize(self):
        """
        Initialize the factor matrices for the ASC update.
        The factor matrices are the absolute of the randn generation. Then:
          The 1st factor gets the sum-to-one constraint.
          The 2nd and 3rd factors are normalized column-wise.
        This should add an artificial channel row to the 2nd factor, which is
          equal to the element-wise inverse of the 3rd factor's last row.
        This should add an artificial channel slab to the tensor, which is
          initialized element-wise with the tensor's mean.
        :return:
        """
        # Initialize the factor matrices
        self.factors0 = [
            np.abs(np.random.randn(dim, self.tensor_rank))
            for dim in self.tensor.shape
        ]

        # The 1st factor gets the sum-to-one constraint
        self.factors0[-2] = normalize(self.factors0[-2], norm="l1", axis=1)

        # The 2nd and 3rd factors are normalized column-wise
        self.factors0[-1] = normalize(self.factors0[-1], norm="l2", axis=0)
        self.factors0[-3] = normalize(self.factors0[-3], norm="l2", axis=0)

        # Add an artificial channel row to the 2nd factor
        delta = np.mean(self.tensor)
        beta = 1e-9
        c_inv = delta / (self.factors0[-3][-1, :] + beta)
        c_inv = c_inv.reshape((1, -1))

        self.factors0[-1] = np.concatenate((self.factors0[-1], c_inv), axis=-2)

        slab = (self.factors0[-3] * c_inv) @ self.factors0[-2].T
        self.tensor = np.concatenate(
            (self.tensor, slab[..., np.newaxis]), axis=-1
        )

    def __restore(self):
        """
        Restore the tensor and factor matrices to their original states.
        """
        self.tensor = self.tensor[..., :-1]
        self.factors[-1] = self.factors[-1][..., :-1]

    def asc_update(
        self, delta: np.ndarray, beta: float = 1e-9,
    ):
        """
        ASC Update
        Assumes the following:
          - The tensor has 3 modes
          - The tensor has been initialized with an artificial channel slice
          - The 2nd factor has been initialized with an artificial channel row
        """
        # Update the artificial channel row in the 2nd factor
        self.factors[-1][-1, :] = delta / (
            self.factors[-3][-1, :].reshape((1, -1)) + beta
        )

        # Update the artificial channel slice in the tensor
        self.tensor[:-1, :, -1] = (
            self.factors[-3][:-1, :] * self.factors[-1][-1, :]
        ) @ self.factors[-2].T

        # Get the tensor unfoldings
        tensor_unfoldings = [
            tl.unfold(self.tensor, mode).T for mode in range(self.tensor_order)
        ]

        return tensor_unfoldings

    def __call__(self, bsum: float = 0):
        self.aoadmmasc(bsum=bsum)

    def aoadmmasc(self, bsum: float = 0):
        # Initialize the factor matrices and the tensor
        self.__initialize()

        # Check if factor matrices are initialized
        assert (
            self.factors0 is not None
        ), "The factor matrices must be initialized"

        # Get the tensor unfoldings, factor matrices, and dual variables
        tensor_unfoldings = [
            tl.unfold(self.tensor, mode).T for mode in range(self.tensor_order)
        ]
        self.factors = self.factors0
        dual_vars = [np.zeros_like(fm) for fm in self.factors0]

        # Loop over the number of iterations
        for itr in range(self.n_iters):
            # Loop over the number of modes (i.e. factor matrices)
            for mode in range(self.tensor_order):
                # Solve the current sub-problem with ADMM
                kr_product = tl.tenalg.khatri_rao(
                    matrices=self.factors[:mode] + self.factors[mode + 1 :]
                )

                self.factors[mode], dual_vars[mode] = self.admms[mode](
                    tensor_unfolding=tensor_unfoldings[mode],
                    kr_product=kr_product,
                    factor=self.factors[mode],
                    dual_var=dual_vars[mode],
                    bsum=bsum,
                )

            # Normalize the columns except for factors of indexes 0 and -2
            # Absorb the weights into the factor matrix of index 0
            weights = la.norm(self.factors[-1][:-1, :], axis=0)
            self.factors[-1][:-1, :] /= weights
            self.factors[0] *= weights

            # Reconstruct the tensor from the estimated factor matrices
            recons_tensor = tl.cp_tensor.cp_to_tensor((None, self.factors))

            # Compare the reconstruction with the original tensor (RMSE)
            self.recons_error[itr] = (
                la.norm(recons_tensor[:, :, :-1] - self.tensor[:, :, :-1])
                / self.tensor_norm
            )

            # # Update the BSUM parameter if necessary
            # bsum = 1e-7  + 0.01 * self.recons_error[itr]

            # Here, it assumes that the data tensor and the factor matrices
            #   are corrected for the ASC constraint
            tensor_unfoldings = self.asc_update(delta=self.tensor_mean,)

        # Restore the tensor and factor matrices to their original states
        self.__restore()

        # Normalize the factors into the diagonal entries
        self.diagonal = la.norm(self.factors[0], axis=0)
        self.factors[0] /= self.diagonal

        # Sort the factors by the diagonal entries in descending order
        diagonal_indices = np.argsort(self.diagonal)[::-1]
        self.diagonal = self.diagonal[diagonal_indices]
        self.factors = [fm[:, diagonal_indices] for fm in self.factors]
