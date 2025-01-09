import functools
import typing as t
from time import time

import numpy as np
from codered.middleware import CodeRedLib
from sage.all import GF, Matrix
from sage.rings.polynomial.multi_polynomial_sequence import (
    PolynomialSequence_generic, PolynomialSequence_gf2)

from ..config import PreprocessingTooHighOutputDim
from ..Experiment import Logger, nolog
from ..helpers import dgv_exact, dimensions, niceprint_mat
from ..helpers.monomials import (fast_coefficients_monomials,
                                 get_all_monomials_fast, matrix_to_psystem)
from .generic import Preprocessing


class LLLReduction(Preprocessing):
    name = "LLL"
    matrix_backend = ("numpy", "sage")

    def __init__(
        self,
        key_bits: int,
        output_dimension: int | None = None,
    ):
        self.output_dim = output_dimension
        self.key_bits = key_bits

    def get_methods(self):
        red: CodeRedLib
        return [
            ("Identity", lambda _: None),
            ("Systematize (Gauss)", lambda red: red.Systematize),
            ("LLL", lambda red: red.LLL),
            (
                "Systematize + LLL",
                lambda red: run_seq(red.Systematize, red.LLL),
            ),
            (
                "Systematize + EpiSort + LLL",
                lambda red: run_seq(red.Systematize, red.EpiSort, red.LLL),
            ),
            (
                "Systematize + EpiSort + LLL + SizeRedBasis",
                lambda red: run_seq(
                    red.Systematize, red.EpiSort, red.LLL, red.SizeRedBasis
                ),
            ),
            (
                "Systematize + EpiSort + LLL + SizeRedBasis + KillTwos",
                lambda red: run_seq(
                    red.Systematize,
                    red.EpiSort,
                    red.LLL,
                    red.SizeRedBasis,
                    red.KillTwos,
                ),
            ),
        ]

    def run(
        self,
        psystem: PolynomialSequence_gf2,
        matrix: np.ndarray,
        monomials,
        additional=None,
        log: Logger = nolog,
    ):
        rng = psystem.ring()
        n = rng.ngens()

        # assert matrix.parent().base_ring().characteristic() == 2
        # import galois
        # time0 = time()
        # matrix_gf2 = log.timefn(matrix.astype)(np.byte, copy=False).view(
        #    galois.GF2
        # )  # In-place view
        # k_, m_ = dimensions(matrix)
        # rank = log.timefn(np.linalg.matrix_rank)(matrix_gf2)
        # if rank < k_:
        #    echelonT = log.timefn(matrix_gf2.T.row_reduce)()
        #    # get row pivots
        #    idx = np.argmax(echelonT, axis=1)
        #    pivot_rows = idx[: np.argmax(idx) + 1]
        #    log.log(f"selecting {len(pivot_rows)} linearly independent rows")
        #    matrix = matrix[pivot_rows, :]  # choose only independent rows

        #    effective_pc = float(len(pivot_rows) / n)
        #    log.log(
        #        f"effectively reducing from PC={len(pivot_rows)}/{n}~{effective_pc}"
        #        " as the rest of rows is linearly dependent"
        #    )
        #    log.data("effective-pc", effective_pc)
        # log.time("independentrows-rank-galois-rowreduce", time() - time0)

        # m4ri matrix rank & row-reduce is much faster than for numpy/galois
        assert len(additional) > 0
        if additional is None or len(additional) == 0:
            # matrix_sage =
            matrix_sage, _ = fast_coefficients_monomials(psystem, backend="sage")
        else:
            matrix_sage, _ = additional[0]

        k, m = matrix_sage.dimensions()
        if log.timefn(matrix_sage.rank)() < k:
            log.log("number of columns < number of rows")
            rowsidx = matrix_sage.pivot_rows()
            matrix = matrix[rowsidx, :]  # remove the rows from numpy matrix
            k = len(rowsidx)
            effective_pc = float(len(rowsidx) / n)
            log.log(
                f"effectively reducing from PC={len(rowsidx)}/{n}~{effective_pc}"
                " as the rest of rows is linearly dependent"
            )
            log.data("effective-pc", effective_pc)

        k, m = dimensions(matrix)
        output_dim = self.output_dim

        if output_dim is not None and k < output_dim:
            raise PreprocessingTooHighOutputDim(
                f"The output dimension {output_dim} > number of linearly independt polynomials {k}."
            )

        log.data("dgv", log.timefn(dgv_exact)(m=m, k=k))

        results = []
        for name, method in self.get_methods():
            red = CodeRedLib(matrix)
            # niceprint_mat(red.P)
            log.timefn(method, name=name)(red)
            # self.info(red, stats_only=True)

            results.append((self.objective_metric(red, output_dim), (name, method)))

        results.sort(key=lambda item: item[0])
        name0, method0 = results[0][1]

        log_results = [
            {"name": name, "metric": metric} for metric, (name, _) in results
        ]
        log.data("methods", log_results, show=False)

        red = CodeRedLib(matrix)
        log.log(f"Applying LLL method '{name0}'")
        log.timefn(method0, name=name0)(red)
        log.data("method", name0)
        metric, new_matrix = log.timefn(self.objective_metric)(
            red, output_dim, output=True
        )
        log.data("metric", float(metric))

        psystem_out = log.timefn(matrix_to_psystem)(
            new_matrix, monomials=monomials, rng=psystem.ring()
        )

        # lll cannot accidentally eliminate some columns - Supp(C) = \sum_i b_i is invariant
        new_monomials = monomials
        return psystem_out, new_matrix, new_monomials

    def objective_metric(
        self, red, target_dim: int | None, output=False
    ) -> np.floating | t.Tuple[np.floating, np.ndarray]:
        """Return metric and preprocessing output for this basis. The lower the better."""
        B = red.B
        k, m = B.shape

        row_metrics = np.array([sum(B[i, :]) for i in range(k)])
        if target_dim is None or target_dim == k:
            # do not search for short rows, output all
            firsts = list(range(k))
        else:
            idxs = np.argsort(row_metrics)
            firsts = idxs[:target_dim]

        if output:
            return np.average(row_metrics[firsts]), B[firsts, :]
        else:
            return np.average(row_metrics[firsts])  # , B[firsts, :]

    def stats(self, red: CodeRedLib):
        B = red.B
        key_bits = self.key_bits
        print("dimensions: ", B.shape)

        k, m = B.shape
        # print("Supp(C):", suppc)
        acc = np.sum(B)
        suppc = len(np.where(red.P[-1, :] == 0)[0])
        print("Supp(C)/sum(|b_i|):", suppc / acc)
        print("avg # of monomials: ", acc / k)

        dists = np.array([self.metric(B[i, :]) for i in range(k)])
        idxdist = np.argsort(dists)
        print(f"shortest {key_bits} weights:", dists[idxdist[:key_bits]])
        print(f"shortest {key_bits} polynomials' weights idxs", idxdist[:key_bits])
        print()
        print(f"shortest weight:", dists[idxdist[0]])
        print(
            f"shortest {key_bits:3d} avg weights:",
            np.average(dists[idxdist[:key_bits]]),
        )
        print(
            f"shortest {2*key_bits:3d} avg weights:",
            np.average(dists[idxdist[: 2 * key_bits]]),
        )
        print(
            f"shortest {4*key_bits:3d} avg weights:",
            np.average(dists[idxdist[: 4 * key_bits]]),
        )
        print(
            f"shortest {8*key_bits:3d} avg weights:",
            np.average(dists[idxdist[: 8 * key_bits]]),
        )
        print()
        print(f"avg first {key_bits} weights:", np.average(dists[:key_bits]))
        print(f"minimal d_min(C) >= ", max(1, dists[0] * 2 ** (-k + 1)))
        # print(f"first {size} weights:", dists[:size])
        # print()
        # print(f"avg longest {size} polynomials' distance:", np.average(dists[idxdist[-size:]]))
        # niceprint(B[idxdist[:5],:])

    def info(self, red, stats_only=True):
        print("-----")
        self.stats(red)

        if not stats_only:
            print("B:")
            niceprint(red.B)  # Print current basis
            print("E:")
            niceprint(red.E)  # Print current Epipodal matrix
        print("profile:", red.l)  # Print current Profile
        print()


def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def compose(*fs):
    return functools.reduce(compose2, fs)


def run_seq(*fs):
    for f in fs:
        f()


def niceprint(B):
    for v in B:
        niceprintv(v)
        # print("".join(["1" if x else "." for x in v]))


def niceprintv(v):
    print("".join(["1" if x else "." for x in v]))
