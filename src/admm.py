from dataclasses import dataclass

import numpy as np
import numpy.linalg as la

from src.proximal_operators import proximal_update_admm


@dataclass
class ADMM:
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

    def __post_init__(self):
        self.factor_names = ["C", "A", "B"]

    def __call__(
        self,
        tensor_unfolding: np.ndarray[tuple[int, int], np.float64],
        kr_product: np.ndarray[tuple[int, int], np.float64],
        factor: np.ndarray[tuple[int, int], np.float64],
        dual_var: np.ndarray[tuple[int, int], np.float64],
        bsum: float = 0,
    ):
        """
        Solves the ADMM sub-problem for a given mode.
        """
        return self.solve(
            tensor_unfolding=tensor_unfolding,
            kr_product=kr_product,
            factor=factor,
            dual_var=dual_var,
            bsum=bsum,
        )

    def solve(
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
        print("")
        print(f"Matrix {self.factor_names[self.tensor_mode]}")

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

            # If the constraint is something like "nonnegative", such that
            #   sometimes the proximal update does not apply any changes (i.e.,
            #   the factor matrix is already nonnegative),
            #   then factor_t and factor are the same, which leads to a
            #   dual variable that remains 0 at the start, corresponding to
            #   a division by zero.
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

            if np.mod(itr, 50) == 0:
                print(f"ADMM Iteration {itr}:"
                      + "\t" +
                      f"Prim Residual: {prim_res:.4e}"
                      + "\t" +
                      f"Dual Residual: {dual_res:.4e}")

            itr += 1

            stop_criter1 = prim_res < self.tol_error
            stop_criter2 = np.isinf(dual_res) or dual_res < self.tol_error
            stop_criter3 = itr >= self.n_iters
            if (stop_criter1 and stop_criter2) or stop_criter3:
                break

        return factor, dual_var
