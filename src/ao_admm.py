from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.linalg as la
import tensorly as tl
from sklearn.preprocessing import normalize

from src.admm import ADMM


@dataclass
class AbstractCPDAOADMM(ABC):

    tensor: np.ndarray[tuple[int, ...], np.float64]
    tensor_rank: int
    admms: list[ADMM, ...]
    n_iters: int

    tensor_order: int = field(init=False)
    tensor_mean: np.ndarray = field(init=False)
    tensor_norm: np.ndarray = field(init=False)
    factors0: list[np.ndarray[tuple[int, int], np.float64], ...] = field(
        init=False
    )
    factors: list[np.ndarray[tuple[int, int], np.float64], ...] = field(
        init=False
    )
    diagonal: np.ndarray[tuple[int], np.float64] = field(init=False)
    recons_error: np.ndarray[tuple[int], np.float64] = field(init=False)

    @abstractmethod
    def __post_init__(self):
        ...

    @abstractmethod
    def _initialize_factors0(self):
        ...

    @abstractmethod
    def _postprocessing_normalize(self):
        ...

    @abstractmethod
    def _postprocessing_sort_columns(self):
        ...

    @abstractmethod
    def __call__(self):
        ...

    @abstractmethod
    def solve(self):
        ...


@dataclass
class CPDAOADMM(AbstractCPDAOADMM):
    def __post_init__(self):
        self.tensor_order = self.tensor.ndim
        self.tensor_mean = np.mean(self.tensor)
        self.tensor_norm = la.norm(self.tensor)
        self.recons_error = np.zeros(self.n_iters)
        self._initialize_factors0()
        self.factors = [
            np.ones((dim, self.tensor_rank)) for dim in self.tensor.shape
        ]
        self.diagonal = np.ones(self.tensor_rank)

    def _initialize_factors0(self):
        # Initialize the factor matrices
        self.factors0 = [
            np.abs(np.random.randn(dim, self.tensor_rank))
            for dim in self.tensor.shape
        ]

        # The 2nd and 3rd factors are normalized column-wise
        self.factors0[-2] = normalize(self.factors0[-2], norm="l2", axis=0)
        self.factors0[-1] = normalize(self.factors0[-1], norm="l2", axis=0)
        self.factors0[-3] = normalize(self.factors0[-3], norm="l2", axis=0)

    def _initialize_solver(self):
        """
        Initialize the factor matrices for the ASC update.
        The factor matrices are the absolute of the randn generation. Then:
          The 1st factor gets the sum-to-one constraint.
          The 2nd and 3rd factors are normalized column-wise.
        :return:
        """
        self._initialize_factors0()

        # Get the tensor unfoldings, factor matrices, and dual variables
        tensor_unfoldings = [
            tl.unfold(self.tensor, mode).T for mode in range(self.tensor_order)
        ]
        dual_vars = [np.zeros_like(fm) for fm in self.factors0]

        return self.factors0, tensor_unfoldings, dual_vars

    def _postprocessing_normalize(self):
        # Normalize the factors into the diagonal entries
        self.diagonal = la.norm(self.factors[-3], axis=0)
        self.factors[-3] /= self.diagonal

    def _postprocessing_sort_columns(self):
        # Sort the factors by the diagonal entries in descending order
        diagonal_indices = np.argsort(self.diagonal)[::-1]
        self.diagonal = self.diagonal[diagonal_indices]
        self.factors = [fm[:, diagonal_indices] for fm in self.factors]

    def __call__(self, bsum: float = 0):
        self.solve(bsum=bsum)

    def solve(self, bsum: float = 0):
        # Initialize
        self.factors, tensor_unfoldings, dual_vars = self._initialize_solver()

        # Loop over the number of iterations
        for itr in range(self.n_iters):
            # print("")
            # temp_str = f"AO Iteration {itr}:"
            # print(temp_str)
            # print("-" * len(temp_str))

            # Loop over the number of modes (i.e. factor matrices)
            for mode in [1, 2, 0]:
                # Solve the current sub-problem with ADMM
                kr_product = tl.tenalg.khatri_rao(
                    matrices=self.factors[:mode] + self.factors[mode + 1 :]
                )

                self.factors[mode], dual_vars[mode] = self.admms[mode].solve(
                    tensor_unfolding=tensor_unfoldings[mode],
                    kr_product=kr_product,
                    factor=self.factors[mode],
                    dual_var=dual_vars[mode],
                    bsum=bsum,
                )

            # Normalize the columns except for factors of indexes 0 and -2
            # Absorb the weights into the factor matrix of index 0
            weights = la.norm(self.factors[-2], axis=0)
            self.factors[-2] /= weights
            weights *= la.norm(self.factors[-1], axis=0)
            self.factors[-1] /= weights
            self.factors[-3] *= weights

            # Reconstruct the tensor from the estimated factor matrices
            recons_tensor = tl.cp_tensor.cp_to_tensor((None, self.factors))

            # Compare the reconstruction with the original tensor (RMSE)
            self.recons_error[itr] = (
                la.norm(recons_tensor - self.tensor)
                / self.tensor_norm
            )

            # Update the BSUM parameter if necessary
            bsum = 1e-7 # + 0.01 * self.recons_error[itr]

        # Normalize the factors into the diagonal entries
        self._postprocessing_normalize()

        # Sort the factors by the diagonal entries in descending order
        self._postprocessing_sort_columns()


@dataclass
class AOADMMASC(CPDAOADMM):
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

    def _initialize_solver(self):
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
        super()._initialize_factors0()

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

        # Get the tensor unfoldings, factor matrices, and dual variables
        tensor_unfoldings = [
            tl.unfold(self.tensor, mode).T for mode in range(self.tensor_order)
        ]
        dual_vars = [np.zeros_like(fm) for fm in self.factors0]

        return self.factors0, tensor_unfoldings, dual_vars

    def _asc_update(
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

    def _restore(self):
        """
        Restore the tensor and factor matrices to their original states.
        """
        self.tensor = self.tensor[..., :-1]
        self.factors[-1] = self.factors[-1][:-1, :]

    def solve(self, bsum: float = 0):
        # Initialize
        self.factors, tensor_unfoldings, dual_vars = self._initialize_solver()

        # Loop over the number of iterations
        for itr in range(self.n_iters):
            print("")
            temp_str = f"AO Iteration {itr}:"
            print(temp_str)
            print("-" * len(temp_str))
            # Loop over the number of modes (i.e. factor matrices)
            for mode in [1, 2, 0]:
                # Solve the current sub-problem with ADMM
                kr_product = tl.tenalg.khatri_rao(
                    matrices=self.factors[:mode] + self.factors[mode + 1 :]
                )

                self.factors[mode], dual_vars[mode] = self.admms[mode].solve(
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

            # Update the BSUM parameter if necessary
            bsum = 1e-7 # + 0.01 * self.recons_error[itr]

            # Here, it assumes that the data tensor and the factor matrices
            #   are corrected for the ASC constraint
            tensor_unfoldings = self._asc_update(delta=self.tensor_mean,)

        # Restore the tensor and factor matrices to their original states
        self._restore()

        # Normalize the factors into the diagonal entries
        self._postprocessing_normalize()

        # Sort the factors by the diagonal entries in descending order
        self._postprocessing_sort_columns()


@dataclass
class AOADMMASCNaive(CPDAOADMM):
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

    def _initialize_solver(self):
        """
        Initialize the factor matrices for the ASC update.
        The factor matrices are the absolute of the randn generation. Then:
          The 1st factor gets the sum-to-one constraint.
          The 2nd and 3rd factors are normalized column-wise.
        :return:
        """
        super()._initialize_factors0()

        # Get the tensor unfoldings, factor matrices, and dual variables
        tensor_unfoldings = [
            tl.unfold(self.tensor, mode).T for mode in range(self.tensor_order)
        ]
        dual_vars = [np.zeros_like(fm) for fm in self.factors0]

        return self.factors0, tensor_unfoldings, dual_vars

    def _asc_update(self):
        self.factors[-2] = normalize(self.factors[-2], norm="l1", axis=1)

    def solve(self, bsum: float = 0):
        # Initialize
        self.factors, tensor_unfoldings, dual_vars = self._initialize_solver()

        # Loop over the number of iterations
        for itr in range(self.n_iters):
            # Loop over the number of modes (i.e. factor matrices)
            for mode in range(self.tensor_order):
                # Solve the current sub-problem with ADMM
                kr_product = tl.tenalg.khatri_rao(
                    matrices=self.factors[:mode] + self.factors[mode + 1 :]
                )

                self.factors[mode], dual_vars[mode] = self.admms[mode].solve(
                    tensor_unfolding=tensor_unfoldings[mode],
                    kr_product=kr_product,
                    factor=self.factors[mode],
                    dual_var=dual_vars[mode],
                    bsum=bsum,
                )

                # Update A with the ASC constraint
                self._asc_update()

            # Normalize the columns except for factors of indexes 0 and -2
            # Absorb the weights into the factor matrix of index 0
            weights = la.norm(self.factors[-1], axis=0)
            self.factors[-1] /= weights
            self.factors[0] *= weights

            # Reconstruct the tensor from the estimated factor matrices
            recons_tensor = tl.cp_tensor.cp_to_tensor((None, self.factors))

            # Compare the reconstruction with the original tensor (RMSE)
            self.recons_error[itr] = (
                la.norm(recons_tensor - self.tensor) / self.tensor_norm
            )

            # # Update the BSUM parameter if necessary
            # bsum = 1e-7 + 0.01 * self.recons_error[itr]

        # Normalize the factors into the diagonal entries
        self._postprocessing_normalize()

        # Sort the factors by the diagonal entries in descending order
        self._postprocessing_sort_columns()
