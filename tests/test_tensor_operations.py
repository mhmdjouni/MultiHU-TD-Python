import time
from pprint import pprint

import tensorly as tl
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from tensorly.metrics import MSE, RMSE


def test_load_fixtures(load_tensor_3, load_factor_matrices_3):

    tensor = load_tensor_3
    fac_mats = load_factor_matrices_3
    rank = fac_mats[0].shape[1]

    pprint(tensor)

    assert tensor.shape == (3, 5, 4)
    assert (fac_mats[0].shape, fac_mats[1].shape, fac_mats[2].shape) == (
        (3, rank),
        (5, rank),
        (4, rank),
    )


def test_tensor_unfolding(load_tensor_3, load_factor_matrices_3):

    tensor = load_tensor_3

    tensor_unfoldings = [
        tl.unfold(tensor, mode=mode).T for mode in range(tensor.ndim)
    ]

    assert (
        tensor_unfoldings[0].shape,
        tensor_unfoldings[1].shape,
        tensor_unfoldings[2].shape,
    ) == ((20, 3), (12, 5), (15, 4),)


def test_khatri_rao(load_tensor_3, load_factor_matrices_3):

    fac_mats = load_factor_matrices_3
    rank = fac_mats[0].shape[1]

    khatri_rao = tl.tenalg.khatri_rao(fac_mats)

    assert khatri_rao.shape == (60, rank)


def test_khatri_rao_mode(load_factor_matrices_3):

    fac_mats = load_factor_matrices_3
    rank = fac_mats[0].shape[1]

    khatri_raos = [
        tl.tenalg.khatri_rao(fac_mats[:mode] + fac_mats[mode + 1 :])
        for mode in range(len(fac_mats))
    ]

    assert (
        khatri_raos[0].shape,
        khatri_raos[1].shape,
        khatri_raos[2].shape,
    ) == ((20, rank), (12, rank), (15, rank),)


def test_khatri_rao_time(load_tensor_3, load_factor_matrices_3):
    tensor = load_tensor_3
    fac_mats = load_factor_matrices_3
    rank = fac_mats[0].shape[1]

    mode = -2
    iters = 1000000
    tensor_unfoldings = [
        tl.unfold(tensor, mode=mode).T for mode in range(tensor.ndim)
    ]
    print('\n')

    start = time.time()
    for i in range(iters):
        kr_product = tl.tenalg.khatri_rao(fac_mats[:mode] + fac_mats[mode + 1:])
        kr_hadamar = kr_product.T @ kr_product
        f = kr_product.T @ tensor_unfoldings[mode]
    time1 = time.time() - start
    print("Using explicit: ", time1)

    start = time.time()
    for i in range(iters):
        kr_hadamar = np.ones((rank, rank))
        for fm in fac_mats[:mode] + fac_mats[mode + 1:]:
            kr_hadamar *= fm.T @ fm
        kr_product = tl.tenalg.khatri_rao(fac_mats[:mode] + fac_mats[mode + 1:])
        f = kr_product.T @ tensor_unfoldings[mode]
    time2 = time.time() - start
    print("Using hadamard: ", time2)


def test_inverse_time(load_tensor_3, load_factor_matrices_3):
    tensor = load_tensor_3
    fac_mats = load_factor_matrices_3
    rank = fac_mats[0].shape[1]

    mode = -2
    iters = 10000
    print('\n')

    kr_product = tl.tenalg.khatri_rao(fac_mats[:mode] + fac_mats[mode + 1:])
    kr_hadamard = kr_product.T @ kr_product
    rho = np.trace(kr_hadamard) / rank
    l = la.cholesky(kr_hadamard + rho * np.eye(rank))
    ll = kr_hadamard + rho * np.eye(rank)

    start = time.time()
    for i in range(iters):
        ll_T_inv1 = la.inv(l.T) @ la.inv(l)
    time1 = time.time() - start
    print(f"Inversion separated: {time1}")

    start = time.time()
    for i in range(iters):
        ll_T_inv2 = la.inv(l @ l.T)
    time2 = time.time() - start
    print(f"Inversion combined: {time2}")

    start = time.time()
    for i in range(iters):
        ll_T_inv3 = la.inv(ll)
    time3 = time.time() - start
    print(f"Inversion with LL_T: {time3}")

    start = time.time()
    for i in range(iters):
        ll_T_inv4 = la.inv(kr_hadamard + rho * np.eye(rank))
    time4 = time.time() - start
    print(f"Inversion without caching: {time4}")

    assert np.allclose(ll_T_inv1, ll_T_inv2)
    assert np.allclose(ll_T_inv1, ll_T_inv3)
    assert np.allclose(ll_T_inv1, ll_T_inv4)

def test_normalize_factor_matrices(load_factor_matrices_3):
    """
    Test the normalization of the factor matrices
    """
    fac_mats = load_factor_matrices_3

    norms = [
        np.linalg.norm(fac_mats[mode], axis=0) for mode in range(len(fac_mats))
    ]

    fac_mats = [fac_mats[mode] / norms[mode] for mode in range(len(fac_mats))]

    norms = [
        np.linalg.norm(fac_mats[mode], axis=0) for mode in range(len(fac_mats))
    ]

    for mode in range(len(fac_mats)):
        assert np.allclose(norms[mode], 1)


def test_normalize_tensor_with_tensorly(load_tensor_3):
    """
    Test the normalization of the factor matrices
    """
    ten1 = load_tensor_3
    iters = 1000000
    print('\n')

    start = time.time()
    for i in range(iters):
        norm_tensorly = tl.norm(ten1)
    time1 = time.time() - start
    print("Tensorly: ", time1)

    start = time.time()
    for i in range(iters):
        norm_numpy = np.linalg.norm(ten1)
    time2 = time.time() - start
    print("Numpy: ", time2)

    start = time.time()
    for i in range(iters):
        norm = np.sum(ten1 ** 2) ** 0.5
    time3 = time.time() - start
    print("Normal: ", time3)

    assert np.allclose(norm_tensorly, norm_numpy)
    assert np.allclose(norm_tensorly, norm)


def test_normalize_factor_matrices_with_tensorly(load_factor_matrices_3):
    """
    Test the normalization of the factor matrices
    """
    fac_mats = load_factor_matrices_3

    fac_mats_cpnorm = tl.cp_tensor.cp_norm((None, fac_mats))
    rec_tens = tl.cp_tensor.cp_to_tensor((None, fac_mats))

    assert fac_mats_cpnorm == tl.norm(rec_tens)

    weights, fac_mats = tl.cp_tensor.cp_normalize((None, fac_mats))

    for mode in range(len(fac_mats)):
        assert np.allclose(np.linalg.norm(fac_mats[mode], axis=0), 1)


def test_partial_normalize_factor_matrices(load_factor_matrices_3):
    """
    Test the normalization of the factor matrices
    """
    fac_mats = load_factor_matrices_3

    weights, fac_mats[1:-1] = tl.cp_tensor.cp_normalize((None, fac_mats[1:-1]))
    fac_mats[-1] *= weights

    for mode in range(1, len(fac_mats) - 1):
        assert np.allclose(np.linalg.norm(fac_mats[mode], axis=0), 1)


def test_tensor_cp_reconstruction(load_factor_matrices_3):
    fac_mats = list(load_factor_matrices_3)
    rank = fac_mats[0].shape[1]
    weights = np.ones((fac_mats[0].shape[1]))

    assert len(fac_mats) == 3
    assert weights.ndim == 1 and weights.size == rank

    reconstructed_tensor = tl.cp_tensor.cp_to_tensor((weights, fac_mats))

    assert reconstructed_tensor.shape == (3, 5, 4)

    weights = None
    reconstructed_tensor = tl.cp_tensor.cp_to_tensor((weights, fac_mats))

    assert reconstructed_tensor.shape == (3, 5, 4)


def test_tensor_norm_from_normalized_reconstrution(load_factor_matrices_3):
    fac_mats = load_factor_matrices_3
    rank = fac_mats[0].shape[1]

    _, fac_mats = tl.cp_tensor.cp_normalize((None, fac_mats))
    reconstructed_tensor = tl.cp_tensor.cp_to_tensor((None, fac_mats))
    norm = tl.norm(reconstructed_tensor)

    print('\n')
    print(norm)

    assert norm < rank


def test_asc_simulation(load_tensor_3, load_factor_matrices_3):
    tensor = load_tensor_3
    fac_mats = load_factor_matrices_3
    rank = fac_mats[0].shape[1]

    delta = np.mean(tensor)

    tensor = np.concatenate((tensor, delta * np.ones((3, 5, 1))), axis=-1)

    assert tensor.shape == (3, 5, 5)

    # This time with a reconstructed tensor
    fac_mats[-2] /= np.sum(fac_mats[-2], axis=1, keepdims=True)

    assert np.allclose(np.sum(fac_mats[-2], axis=1), 1)

    tensor = tl.cp_tensor.cp_to_tensor((None, fac_mats))
    delta = np.mean(tensor)

    beta = 1e-9
    temp = 1.0 / (fac_mats[-3][-1, :] + beta)
    temp = temp.reshape((1, -1))

    fac_mats[-1] = np.concatenate((fac_mats[-1], delta * temp), axis=0)
    recten = tl.cp_tensor.cp_to_tensor((None, fac_mats))

    assert fac_mats[-1].shape == (5, rank)
    assert recten.shape == (3, 5, 5)
    assert np.allclose(recten[-1, :, -1], delta)


def test_asc_initialize(load_factor_matrices_3):
    # Initialize the factor matrices
    factors = load_factor_matrices_3
    tensor = tl.cp_tensor.cp_to_tensor((None, factors))
    rank = factors[0].shape[1]

    # The 1st factor gets the sum-to-one constraint
    factors[-2] = normalize(factors[-2], norm='l1', axis=1)

    # The 2nd and 3rd factors are normalized column-wise
    factors[-1] = normalize(factors[-1], norm='l2', axis=0)
    factors[-3] = normalize(factors[-3], norm='l2', axis=0)

    # Add an artificial channel row to the 2nd factor
    delta = np.mean(tensor)
    beta = 1e-9
    c_inv = delta / (factors[-3][-1, :].reshape((1, -1)) + beta)
    # c_inv = c_inv.reshape((1, -1))

    factors[-1] = np.concatenate((factors[-1], c_inv), axis=-2)

    # I want the shape (K, I, 1)
    slab = (factors[-3] * c_inv) @ factors[-2].T
    tensor = np.concatenate((tensor, slab[..., np.newaxis]), axis=-1)

    assert factors[-1].shape == (5, rank)
    assert tensor.shape == (3, 5, 5)
    assert np.allclose(tensor[-1, :, -1], delta)

    factors[-3], factors[-2], _ = tuple(load_factor_matrices_3)

    # Update the artificial channel row in the 2nd factor
    factors[-1][-1, :] = delta / (factors[-3][-1, :].reshape((1, -1)) + beta)

    # Update the artificial channel slice in the tensor
    tensor[:-1, :, -1] = (factors[-3][:-1, :] * factors[-1][-1, :]) @ factors[-2].T

    # Get the tensor unfoldings
    tensor_unfoldings = [
        tl.unfold(tensor, mode).T for mode in range(tensor.ndim)
    ]


def test_asc_update(load_factor_matrices_3):
    # Initialize the factor matrices
    factors = load_factor_matrices_3
    tensor = tl.cp_tensor.cp_to_tensor((None, factors))
    rank = factors[0].shape[1]

    # The 1st factor gets the sum-to-one constraint
    factors[-2] = normalize(factors[-2], norm='l1', axis=1)

    # The 2nd and 3rd factors are normalized column-wise
    factors[-1] = normalize(factors[-1], norm='l2', axis=0)
    factors[-3] = normalize(factors[-3], norm='l2', axis=0)

    # Add an artificial channel row to the 2nd factor
    delta = np.mean(tensor)
    beta = 1e-9
    c_inv = delta / (factors[-3][-1, :].reshape((1, -1)) + beta)
    # c_inv = c_inv.reshape((1, -1))

    factors[-1] = np.concatenate((factors[-1], c_inv), axis=-2)

    # I want the shape (K, I, 1)
    slab = (factors[-3] * c_inv) @ factors[-2].T
    tensor = np.concatenate((tensor, slab[..., np.newaxis]), axis=-1)

    factors[-3], factors[-2], _ = tuple(load_factor_matrices_3)

    assert factors[-1].shape == (5, rank)
    assert tensor.shape == (3, 5, 5)
    assert np.allclose(tensor[-1, :, -1], delta)

    # Update the artificial channel row in the 2nd factor
    factors[-1][-1, :] = np.ones((1, rank))

    # Update the artificial channel slice in the tensor
    # slab = (factors[-3][:-1, :] * factors[-1][-1, :]) @ factors[-2].T
    tensor[:-1, :, -1] = np.ones((2, 5))

    # Get the tensor unfoldings
    tensor_unfoldings = [
        tl.unfold(tensor, mode).T for mode in range(tensor.ndim)
    ]

    assert factors[-1].shape == (5, rank)
    assert tensor.shape == (3, 5, 5)
    assert np.allclose(tensor[-1, :, -1], delta)


def test_sklearn_mse_tensor(load_tensor_3, load_factor_matrices_3):
    tensor1 = load_tensor_3
    factors = load_factor_matrices_3
    tensor2 = tl.cp_tensor.cp_to_tensor((None, factors))

    e_mse_implicit = MSE(tensor1, tensor2)
    e_mse = np.sum(np.square(tensor1 - tensor2)) / tensor1.size

    e_rmse_implicit = RMSE(tensor1, tensor2)
    e_rmse = np.sqrt(np.sum(np.square(tensor1 - tensor2))) / np.sqrt(tensor1.size)

    e_error_tensorly = tl.norm(tensor1 - tensor2)
    e_error_numpy = np.linalg.norm(tensor1 - tensor2)
    e_error = np.sqrt(np.sum(np.square(tensor1 - tensor2)))


def test_argsort(load_factor_matrices_3):
    factors = load_factor_matrices_3

    diagonal, factors = tl.cp_tensor.cp_normalize((None, factors))

    for i in range(len(factors)):
        assert np.allclose(np.linalg.norm(factors[i], axis=0), 1)

    indices = np.argsort(diagonal)[::-1]
    diagonal = diagonal[indices]
    factors = [factor[:, indices] for factor in factors]

    for i in range(len(factors)):
        assert np.allclose(np.linalg.norm(factors[i], axis=0), 1)
