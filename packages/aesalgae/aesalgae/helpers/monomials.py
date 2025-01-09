import math
import multiprocessing as mp
import operator
import typing as t
from functools import reduce
from time import time

import numpy as np
from sage.all import (GF, BooleanPolynomialRing, Integer, Matrix, TermOrder,
                      dimension)
from sage.modules.free_module_element import vector
from sage.parallel.decorate import parallel
from sage.rings.polynomial.multi_polynomial_sequence import (
    PolynomialSequence, PolynomialSequence_gf2)
from sage.rings.polynomial.pbori.pbori import BooleanMonomialMonoid, BooleSet
from sage.structure.element import Matrix as Matrix_t

from ..config import MAX_CORES
from ..Experiment import Logger, nolog
from ..helpers import dimensions, niceprint_mat
from .helpers import batched_it


def argsort(seq, reverse=False):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def fast_coefficients_monomials(
    psystem: PolynomialSequence_gf2,
    sparse=False,
    order: TermOrder | None = "degrevlex",
    backend: str = "sage",
    log: Logger = nolog,
) -> t.Tuple[Matrix, vector]:
    """
    Construct a coefficient matrix from polynomial system psystem in parallel.

    If ord = None, keep original ring monomial order. If not None, construct the matrix
    in new ordering by first determining indices via sorting ring monomials wrt new order.

    This should be faster than default sage impl because we do not need to first convert all
    polys to the new ordering, just one. And it is parallelized.

    Parts augmented from original PolynomialSequence_gf2.coefficients_monomials
    """
    assert backend in ["sage", "numpy"], f"invalid backend {backend}"
    old_rng = psystem.ring()
    new_rng = old_rng.clone().change_ring(order=order)
    new_monom_base = BooleanMonomialMonoid(new_rng)

    # could be possibly sped up by considering all possible monomials instead only those present
    # ~ 1.3 s SR(3,2,2,4)
    orig_monoms = log.timefn(get_all_monomials_fast)(psystem)
    len_mon = len(orig_monoms)

    # timings:
    # map to new, sort, map back: 8.3 s
    # argsort: 5.5 s

    # [3, 2, 0, 1] -> [0, 1, 2, 3]; 3->0, 2->1, 0->2, 1->3 and convert keys to old
    # monomials
    # ~ 0.9 s
    # new_ord_monoms = sorted(map(new_monom_base, orig_monoms), reverse=True)
    # ~ 0.6 s
    # old_monoms_in_new_order = list(map(old_rng, new_ord_monoms))
    # return psystem.coefficients_monomials(order=old_monoms_in_new_order, sparse=sparse)

    time0 = time()
    # ~ 0.8 s
    new_ord_monoms = list(map(new_monom_base, orig_monoms))
    log.time("monomials-new-ordering", time() - time0)
    # ~ 0.1 s
    time0 = time()
    new_ord_monoms_idx = argsort(new_ord_monoms, reverse=True)
    log.time("monomials-sort", time() - time0)
    # ~ 0.1 s
    old_v = [orig_monoms[idx] for idx in new_ord_monoms_idx]

    # keys must be monomials in the old ring
    idx_ord = dict(zip(old_v, range(len_mon)))

    if backend == "numpy":
        assert not sparse
        numpy_A = log.timefn(_numpy_psystem_to_matrix)(psystem, len_mon, idx_ord)
        # print("numpy")
        # niceprint_mat(numpy_A)
        return numpy_A, old_v

    # matrix over GF(2)
    base_rng = psystem.ring().base_ring()
    A = Matrix(base_rng, len(psystem), len_mon, sparse=sparse)  # over GF2
    one = old_rng.one()

    @parallel(ncpus=MAX_CORES)
    def output_monom_vector(i, poly):
        # parallel does not mutate global state
        vec = Matrix(base_rng, 1, len_mon, sparse=sparse)
        for m in poly:
            vec[0, idx_ord[m]] = one
        return vec

    time0 = time()
    rows = sorted(
        list(
            output_monom_vector(list(enumerate(psystem))),
        )
    )  # sort by i
    time1 = time()
    log.time("parallel_polynomials-monomial-row", time1 - time0)
    for i, row in enumerate(rows):
        A[i, :] = row[1]  # extract result
    time2 = time()
    log.time("constructed-full-matrix", time2 - time1)

    # niceprint_mat(A[:, :20])
    # print(v)
    # print("sage")
    # niceprint_mat(A)
    return A, old_v  # vector(v)

    # for i, poly in enumerate(psystem):
    #    # write_monoms(i, poly)
    #    for m in poly:
    #        A[i, idx_ord[m]] = one
    # return A, vector(v)

    # first_monom_old = list(sorted(orig_monoms, reverse=True))[0]
    # first_monom_new = new_ord_monoms[0]
    # print(first_monom_old)
    # print(first_monom_new)
    # first_monom_old == first_monom_new
    # > k000*k001*k002*k003*k010*k011*k012*k013*k020*k021*k022*k023*k030*k031*k032*k033
    # > k033*k032*k031*k030*k023*k022*k021*k020*k013*k012*k011*k010*k003*k002*k001*k000
    # > False


def get_all_monomials_fast(pseq: PolynomialSequence_gf2) -> tuple:
    """Get all monomials present in pseq."""

    M = pseq.ring().zero().set()
    for f in pseq:
        M = M.union(f.set())
    return tuple(M)
    # orig_monoms = log.timefn(psystem.monomials, "get all monomials")()


def get_density_fast(pseq: PolynomialSequence_gf2) -> float:
    """Get average number of monomials in each polynomial out of all present."""

    # key_bits = pseq.ring().ngens()
    len_present_monoms = len(get_all_monomials_fast(pseq))
    k = sum(len(part) for part in pseq.parts())
    s = sum(Integer(len(f.set())) for f in pseq)
    return float(s / (k * len_present_monoms))

    # avg = 0
    # key_bits = pseq.ring().ngens()
    # for f in pseq:
    #    avg += len(f.set()) / (2**key_bits)
    # return avg
    # orig_monoms = log.timefn(psystem.monomials, "get all monomials")()


# def matrix_and_monomials(to_backend: str, from_backend: str|None=None, psystem=None,  matrix=None, monomials=None, **kwargs):
#    """ does not convert matrices between backends """
#    assert to_backend in ["sage", "numpy"]
#    if from_backend is None:
#       assert from_backend in ["sage", "numpy"]
#
#    if matrix is None:
#        assert psystem is not None
#        return fast_coefficients_monomials(
#            psystem,
#            backend=to_backend,
#            **kwargs
#        )
#
#    if from_backend == to_backend:
#       return matrix, monomials
#
#    assert psystem is not None


def matrix_to_psystem(
    matrix: Matrix_t | np.ndarray, monomials, rng: BooleanPolynomialRing
):
    """Convert monomial matrix to polynomial sequence"""

    # mon_vec = log.timefn(vector)(monomials)
    # out_psystem = new_matrix * mon_vec # slow, ~10 s for SR(3,2,2,4) output dim =4

    # ~ 5s SR(3,2,2,4), 5->4 PC pairs monomelim
    # @parallel(ncpus=MAX_CORES)
    # def sum_polynomials(i):
    #    return sum(combinations[i, j] * psystem[j] for j in range(k))

    zero = rng.zero()
    k, m = dimensions(matrix)

    # possible to optimize by constructing packed representation from entire row for
    # numpy and then coercing to polynomial
    @parallel(ncpus=MAX_CORES)
    def construct_polynomial(i):
        M = zero.set()
        for j, m in enumerate(monomials):
            if matrix[i, j]:
                M = M.union(m.set())
        return rng(M)  # coerce to polynomial

    out_psystem = list(
        map(
            lambda x: x[1],
            sorted(
                list(construct_polynomial(list(range(k)))),
            ),
        )
    )
    return PolynomialSequence(out_psystem)


def _numpy_psystem_to_matrix_shared(psystem, len_monomials, monom_idx):
    """Construct a numpy matrix by creating a shared memory and letting all the
    processes mutate the shared memory; hence filling the array in place. This works as
    each process controls different parts of the matrix.

    For some reason, the computation hangs when the matrix is craeted by this function when computing LLL.
    """

    def shared_to_numpy(shared_arr, dtype, shape):
        """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
        No copy is involved, the array reflects the underlying shared buffer."""
        return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)

    def np_create_shared_array(dtype, shape):
        """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
        Note that the buffer values are not initialized.
        """
        dtype = np.dtype(dtype)
        # Get a ctype type from the NumPy dtype.
        cdtype = np.ctypeslib.as_ctypes_type(dtype)
        # Create the RawArray instance.
        size = reduce(operator.mul, shape, int(1))
        shared_arr = mp.RawArray(cdtype, size)
        # Get a NumPy array view.
        arr = shared_to_numpy(shared_arr, dtype, shape)
        return shared_arr, arr

    dtype = np.bool_
    k, m = len(psystem), len_monomials
    shape = (k, m)
    shared_arr, arr = np_create_shared_array(dtype, shape)
    # zero out not needed
    arr.flat[:] = np.zeros(k * m)

    # for m in poly:
    #    vec[0, idx_ord[m]] = one
    # return vec

    shared_arr_wparams = (shared_arr, dtype, shape)
    one = int(1)

    @parallel(ncpus=24)
    def populate_arr(i0, i1):
        arr = shared_to_numpy(*shared_arr_wparams)
        for i in range(i0, i1):
            for m in psystem[i]:
                arr[i, monom_idx[m]] = one

    # Create a Pool of processes and expose the shared array to the processes, in a global variable
    # (_init() function).
    nproc = MAX_CORES
    chunk = k // nproc
    if chunk == 0:
        _ = list(populate_arr([(ip, ip + 1) for ip in range(k)]))
    else:
        _ = list(
            populate_arr(
                [(ip, min(ip + chunk, k)) for ip in range(math.ceil(k / chunk))]
            )
        )
    # Show [0, 1, 2, 3...]
    return arr


def _numpy_psystem_to_matrix(psystem, len_monomials, monom_idx):
    """ """
    k, m = len(psystem), len_monomials
    shape = (k, m)
    arr = np.zeros((k, m), dtype=np.bool_)

    # for m in poly:
    #    vec[0, idx_ord[m]] = one
    # return vec

    one = int(1)

    @parallel(ncpus=24)
    def populate_arr(i0, i1):
        my_arr = np.zeros((i1 - i0, m))
        for i in range(i1 - i0):
            for mon in psystem[i0 + i]:
                my_arr[i, monom_idx[mon]] = one

        return my_arr

    nproc = MAX_CORES
    chunk = k // nproc
    if chunk == 0:
        res = list(populate_arr([(ip, ip + 1) for ip in range(k)]))
    else:
        res = list(
            populate_arr(
                [
                    (ip * chunk, min((ip + 1) * chunk, k))
                    for ip in range(math.ceil(k / chunk))
                ]
            )
        )

    for inp, out in res:
        args, _ = inp
        i0, i1 = args
        arr[i0:i1, :] = out

    return arr


# 100 th
# k=250
# chunk=2
# (i, i+2) ... i=0...125


###
### SCRATCH multiprocessing
### one needs to create global shared variables from multiprocessing primitives
### and there is no way to provide access
# def construct_matrix_np():
#    """Construct a numpy matrix by creating a shared memory and letting all the
#    processes mutate the shared memory; hence filling the array in place. This works as
#    each process controls different parts of the matrix.
#    """
#
#    def _init(shared_arr_, psystem_):
#        # The shared array pointer is a global variable so that it can be accessed by the
#        # child processes. It is a tuple (pointer, dtype, shape).
#        global gShared_arr, gPsystem
#        gShared_arr = shared_arr_
#        gPsystem = psystem_
#
#    def shared_to_numpy(shared_arr, dtype, shape):
#        """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
#        No copy is involved, the array reflects the underlying shared buffer."""
#        return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)
#
#    def np_create_shared_array(dtype, shape):
#        """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
#        Note that the buffer values are not initialized.
#        """
#        dtype = np.dtype(dtype)
#        # Get a ctype type from the NumPy dtype.
#        cdtype = np.ctypeslib.as_ctypes_type(dtype)
#        # Create the RawArray instance.
#        size = reduce(operator.mul, shape, int(1))
#        shared_arr = mp.RawArray(cdtype, size)
#        # Get a NumPy array view.
#        arr = shared_to_numpy(shared_arr, dtype, shape)
#        return shared_arr, arr
#
#    @parallel(ncpus=MAX_CORES)
#    def populate_arr(idx_range):
#        i0, i1 = idx_range
#        arr = shared_to_numpy(*gShared_arr)
#        arr[i, i] = i
#
#    ######
#    ######
#    def driver():
#        # For simplicity, make sure the total size is a multiple of the number of processes.
#        n_processes = MAX_CORES
#        N = int(10000 // n_processes)
#        assert N % n_processes == 0
#
#        # Initialize a shared array.
#        dtype = np.int32
#        shape = (N, N)
#        shared_arr, arr = np_create_shared_array(dtype, shape)
#        arr.flat[:] = np.zeros(N * N)
#        # Show [0, 0, 0, ...].
#        # print(arr)
#
#        # Create a Pool of processes and expose the shared array to the processes, in a global variable
#        # (_init() function).
#        with mp.Pool(
#            n_processes, initializer=_init, initargs=((shared_arr, dtype, shape),)
#        ) as p:
#            n = N // n_processes
#            # Call parallel_function in parallel.
#            p.map(parallel_function, [(k * n, (k + 1) * n) for k in range(n_processes)])
#        # Close the processes.
#        p.join()
#        # Show [0, 1, 2, 3...]
#        print(arr)
