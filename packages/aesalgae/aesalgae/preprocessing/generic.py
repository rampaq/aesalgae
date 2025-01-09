import typing as t
from time import time

import numpy as np
from aesalgae.helpers.helpers import dimensions
from aesalgae.helpers.monomials import (fast_coefficients_monomials,
                                        get_all_monomials_fast)
from sage.rings.polynomial.multi_polynomial_sequence import \
    PolynomialSequence_generic
from sage.structure.element import Matrix

from ..Experiment import Logger, nolog


class Preprocessing:
    name: str
    matrix_backend: str | tuple[str] | None = None

    def run(
        self,
        psystem: PolynomialSequence_generic,
        matrix: Matrix | None,
        monomials: list | None,
        additional=None,
        log: Logger = nolog,
    ) -> t.Tuple[PolynomialSequence_generic, Matrix | None, list | None]:
        """
        matrix: rows correspond to polynomials in psystem and columns are indexed by `monomials` and entries are coefficients
        """
        ...

    def get_backend(self) -> tuple[str] | None:
        if self.matrix_backend is None:
            return None
        if isinstance(self.matrix_backend, str):
            return (self.matrix_backend,)
        return self.matrix_backend


class PreprocessingPipe(Preprocessing):
    pipe: list[Preprocessing]
    name: str

    def __init__(self, *args, collect_stats: t.Callable | None = None):
        """args is a sequence of Preprocessing instances"""
        assert len(args) > 0
        self.collect_stats = collect_stats
        self.name = ">>".join([preproc.name for preproc in args])
        self.pipe = list(args)

    def prepare_matrices(
        self, psystem, i: int = 0, only_additional=False, log: Logger = nolog
    ) -> tuple[Matrix, tuple, list[tuple[Matrix, tuple]]]:
        """for_i: for preprocessing with index i (default 0)"""
        assert 0 <= i < len(self.pipe)

        backend = self.pipe[i].get_backend()

        matrix, monomials, additional_matrices = None, None, []
        if backend is not None:
            if isinstance(backend, str):
                backend = (backend,)
            if not only_additional:
                matrix, monomials = log.timefn_log(fast_coefficients_monomials)(
                    psystem, backend=backend[0]
                )
            for backend_additional in backend[1:]:
                matrix0, monomials0 = log.timefn_log(fast_coefficients_monomials)(
                    psystem, backend=backend_additional
                )
                additional_matrices.append((matrix0, monomials0))
        return matrix, monomials, additional_matrices

    def run(
        self,
        psystem,
        matrix=None,
        monomials=None,
        additional=None,
        log: Logger = nolog,
    ):
        if additional is None:
            additional = []
        for i, preproc in enumerate(self.pipe):
            backend = preproc.get_backend()
            if backend and backend[0] == "numpy" and not isinstance(matrix, np.ndarray):
                matrix, monomials = None, None  # zero out

            if backend and (
                matrix is None
                or (
                    isinstance(backend, tuple)
                    and len(backend) > len(additional) + int(matrix is not None)
                )
            ):
                # matrix not created or mismatched format
                only_additional = matrix is not None
                if only_additional:
                    _, _, additional = self.prepare_matrices(
                        psystem, i, only_additional=True, log=log
                    )
                else:
                    matrix, monomials, additional = self.prepare_matrices(
                        psystem, i, only_additional=False, log=log
                    )
                print(dimensions(matrix))

            if self.collect_stats:
                self.collect_stats(psystem, matrix, monomials, preproc.name)

            log.log(f"chaining {preproc.name}")
            psystem, matrix, monomials = log.timefn_log(preproc.run, name=preproc.name)(
                psystem,
                matrix,
                monomials,
                additional=additional,
                log=log.sublogger(preproc.name),
            )
            log.log(f"finished {preproc.name}")

        return psystem, matrix, monomials
