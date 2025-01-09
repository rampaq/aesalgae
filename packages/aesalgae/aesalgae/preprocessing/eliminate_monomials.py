import bisect
import time as time_main
import typing as t
from collections import defaultdict
from functools import reduce
from time import time

import numpy as np
from sage.all import Matrix, parallel, vector
from sage.rings.polynomial.multi_polynomial_sequence import (
    PolynomialSequence, PolynomialSequence_generic, PolynomialSequence_gf2)
from sage.rings.polynomial.pbori.pbori import BooleanPolynomialIdeal

from ..config import MAX_CORES, PreprocessingTooHighOutputDim
from ..Experiment import Logger, nolog
from ..helpers import niceprint_mat
from ..helpers.monomials import (fast_coefficients_monomials,
                                 get_all_monomials_fast, matrix_to_psystem)
from .generic import Preprocessing


class MonomialElimination(Preprocessing):
    name = "MonomialElimination"
    matrix_backend = "sage"

    start_l: int | None
    output_dim: int
    output_progress: bool

    def __init__(self, output_dim: int = 1):  # , strict_output_dim: bool = True):
        """
        output_dim : minimal required dimension of subcode C'; the algorithm will find as
            high l as possible so that this constraint is satisfied

        strict_output_dim:
            when True (default), throw error if output dimension cannot be satisfied
            when False : if output dimension cannot be satisfied, try to produce
            only linear equations. When this fails, throw error.
        """

        self.output_dim = output_dim
        assert self.output_dim > 0

    #        self.strict_output_dim = strict_output_dim

    # super().__init__()

    def compute_combinations(
        self,
        B: Matrix,
        output_dim: int,
        start_l: int | None = None,
        lmax: int | None = None,
        log: Logger = nolog,
    ):  # , poly_seq: PolynomialSequence_gf2):
        """
        Compute subcode C' of original code C with basis B such that first l columns of C' are zeros.

        MAKE SURE THAT B IS NOT IN SPARSE FORMAT WHEN NOT SPARSE! OTHERWISE, MULTIPLICATION TAKES AGES.

        Make sure input matrix has linearly independent rows.

        If output_dim > k, then you can override the discrepancy by specifying lmax. If
        elimination all the way to lmax is possible, then output the corresponding
        combinations. This is useful when suspecting that one can eliminate all the way
        to linear (affine) terms in the original polynomials. This is achieved when lmax
        is set to the index of first non-linear monomial - 1.
        - If it is not possible to eliminate lmax (corresponding kernel has dimension 0),
        throw error.

        B: sage matrix from F_2^{k x m}
        lmax: maximum number of columns to eliminate. Used primarily to prevent
            reducing the system to 0=0, so max_l should be set to (number of all monomials
            in system) - (linear and constant monomials in system). If not specified, set to m-1.

        Returns:
            matrix M from F_2^{output_dim x k} such that M*B = basis of code C'
            in other words, (b_i)' = sum_{j=1}^k m_ij b_j where b_j are input polynomials and b_i' output polynomials
            #if output_progress=True, return subcodes, (ls, dims)
        """

        m: int
        k, m = B.dimensions()
        assert m >= k
        assert k >= output_dim

        wanted_dimker = output_dim
        if lmax is None:
            lmax = m - 1
        wanted_rank = k - wanted_dimker

        if start_l is None:
            l0 = wanted_rank
        else:
            l0 = start_l
        # set l0 and try bisecting values
        # l is the final value of number of eliminable columns leading to
        # output_dimensional dimker
        l = (
            bisect.bisect(
                range(m), wanted_rank, lo=l0, hi=lmax, key=lambda l: B[:, :l].rank()
            )
            - 1
        )
        log.data("eliminated-monoms", l)
        assert l >= 0 and l <= lmax

        # left kernel
        time1 = time()
        dimker = k - B[:, :l].rank()
        kermat = B[:, :l].T.right_kernel_matrix()[:dimker]
        log.time(f"{l}-ker", time() - time1)
        return l, kermat

    def run(
        self,
        psystem: PolynomialSequence_generic,
        matrix: Matrix,
        monomials,
        additional=None,
        log: Logger = nolog,
    ):
        rng = psystem.ring()
        n = rng.ngens()

        # make sure input is linearly independent
        k, m = matrix.dimensions()
        if matrix.rank() < k:
            log.log("number of columns < number of rows")
            rowsidx = matrix.pivot_rows()
            matrix = matrix[rowsidx, :]
            k = len(rowsidx)
            effective_pc = float(len(rowsidx) / n)
            log.log(
                f"effectively reducing from PC={len(rowsidx)}/{n}~{effective_pc}"
                " as the rest of rows is linearly dependent"
            )
            log.data("effective-pc", effective_pc)

        output_dim = self.output_dim
        if output_dim is None:
            output_dim = k
        if k < self.output_dim:
            raise PreprocessingTooHighOutputDim(
                f"The output dimension {output_dim} > number of linearly independt polynomials {k}."
            )

        l, combinations = log.timefn_log(self.compute_combinations)(
            matrix, output_dim=output_dim
        )
        dim, k_ = combinations.dimensions()
        assert k == k_

        time0 = time()
        new_matrix = combinations * matrix
        log.time("matrix multiplication", time() - time0)

        new_monomials = monomials[l:]
        out_psystem = log.timefn(matrix_to_psystem)(new_matrix, monomials, rng)

        return out_psystem, new_matrix, new_monomials
