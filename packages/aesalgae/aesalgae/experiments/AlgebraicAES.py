import typing as t
from time import time

import numpy as np
from aesalgae.helpers.helpers import niceprint_mat
from aesalgae.preprocessing.generic import PreprocessingPipe
from saes.saes import AES
from sage.all import RDF, Ideal
from sage.all import log as log_sage
from sage.interfaces.magma import Magma, magma
from sage.rings.polynomial.multi_polynomial_sequence import (
    PolynomialSequence, PolynomialSequence_generic, PolynomialSequence_gf2)
from sage.rings.polynomial.pbori.pbori import BooleanPolynomialIdeal
from sage.structure.element import Matrix

from ..config import (MAX_CORES, MagmaGroebnerBasisCrash,
                      MagmaGroebnerBasisInvalid, PreprocessingParamsInvalid)
from ..Experiment import Experiment, Logger, nolog
from ..groebner import (degree_of_semiregularity_gf2, fgb_gb_solve,
                        get_num_solutions_from_gb, is_key_in_ideal,
                        magma_gb_solve)
from ..helpers import dimensions
from ..helpers.monomials import (fast_coefficients_monomials,
                                 get_all_monomials_fast, get_density_fast)
from ..preprocessing.eliminate_monomials import MonomialElimination
from ..preprocessing.inflate import InflateSystem
from ..preprocessing.lll import LLLReduction

SPARSITY_TRESHOLD = 0.3
key = None


class AlgebraicAES(Experiment):
    aes_nrce: t.Tuple[int, int, int, int]
    key_bits: int
    pc_pairs: int

    def __init__(
        self,
        n: int,
        r: int,
        c: int,
        e: int,
        pc_pairs: int,
        reduced_dim: int | None = None,
        gb_method: str = "magma",
        preprocessing: str | None = None,
        sparse: bool | None = None,
        collect_statistics: bool = False,
        enable_gpu: bool = True,
        num_threads: int = 1,
        performance_mode: bool = False,
        magma_interface: Magma | None = None,
    ):
        """
        SR(n,r,c,e) without auxilary variables - just key variables

        Args:
            pc_pairs: number of systems to generate for different plaintexts and same key
            sparse: compute with sparse matrices? set to None for automatic decisions
            collect_statistics:
                Collect statistics about the system. It might take some time for larger
                systems. When False, some automatic decision making is inactive.
            reduced_dim: target number of polynomials resulting from preprocessing in
                multiples of key bits; i.e. for reduced_dim=5 and SR(3,2,2,4),
                preprocessing outputs 5*16 polynomials. When None (default), reduced_dim=pc_pairs.
            magma_interface: custom magma interface in case you want to supply e.g. a
                different magma executable
            enable_gpu: enable GPU (only applicable for dense variant of magma), default True. When crashes occur, turn it off
            enable_threads: True to enable threaded GB computation with `num_threads` threads if None, threads = not enable_gpu
            num_threads: if threads for Mamga are enabled, use this many
            performance_mode: disable collecting statistics not used in computations


        """
        super().__init__(name="AES")

        self.aes_nrce = (n, r, c, e)
        self.pc_pairs = pc_pairs
        self.reduced_dim = reduced_dim if reduced_dim is not None else pc_pairs
        self.key_bits = r * c * e
        self.sparse = sparse
        self.collect_statistics = collect_statistics
        self.enable_gpu = enable_gpu
        self.gb_method = gb_method
        self.magma = magma_interface if magma_interface is not None else magma
        self.num_threads = num_threads
        self.is_telemetry_enabled = not performance_mode
        self.monomial_matrix_backend = None

        if self.num_threads > 1 and self.enable_gpu:
            self.log.log(
                "WARNING: using multiple threads and GPU might lead to performance decrease according to general Magma manual."
                " According to GroebnerBasis parallelism manual, Magma will choose automatically so it should not matter?"
            )

        prepipe = preprocessing.split(",") if preprocessing is not None else []
        # assert set(prepipe) <= set(["lll", "monomelim", "inflate"])

        bases = []
        max_reduce_dim = self.pc_pairs * self.key_bits
        do_not_reduce_dimension = reduced_dim is None
        info = []
        for preproc in prepipe:
            if preproc.startswith("inflate"):
                shuffle, inv, custom = False, False, False
                args = preproc[len("inflate") + 1 : -1]
                if args == "shuffle":
                    shuffle = True
                elif args == "inv":
                    inv = True
                elif args == "custom":
                    custom = True
                base = InflateSystem(shuffle=shuffle, inverse=inv, custom=custom)
                inflation = base.inflation_factor(self.key_bits)
                max_reduce_dim *= inflation
                info.append(f"InflateSystem(inflation_factor={inflation}, inv={inv})")
            elif preproc == "lll":
                # None = do not reduce dimension and just do LLL
                out_dim = None if do_not_reduce_dimension else reduced_dim
                base = LLLReduction(
                    key_bits=self.key_bits,
                    output_dimension=out_dim,
                )
                info.append(
                    f"LLL(output_dimension={str(out_dim) if out_dim else "keep"})"
                )
                do_not_reduce_dimension = True
            elif preproc == "monomelim":
                if (
                    do_not_reduce_dimension
                    or reduced_dim is None
                    or reduced_dim >= max_reduce_dim
                ):
                    raise PreprocessingParamsInvalid(
                        "Cannot apply MonomialElimination without reduction."
                    )
                else:
                    base = MonomialElimination(output_dim=reduced_dim)
                    info.append(f"MonomialElimination(output_dimension={reduced_dim})")
                    do_not_reduce_dimension = True
            else:
                print(preproc)
                assert False
            bases.append(base)
        self.preprocessing_info = " -> ".join(info)
        if prepipe:
            self.preprocessing_base = PreprocessingPipe(*bases)
            self.do_preprocessing = True
        else:
            self.preprocessing_base = None
            self.do_preprocessing = False

        assert self.pc_pairs <= 2**self.key_bits

    def run(self):
        log = self.log
        n, r, c, e = self.aes_nrce
        pc_pairs = self.pc_pairs
        key_bits = self.key_bits

        print(f"AES SR({n}, {r}, {c}, {e}), key={key_bits} bits, pc_pairs={pc_pairs}")

        psystem, key = log.timefn_log(self.generate_aes)()
        self.key = key

        # A, monomials = None, None
        # if self.do_preprocessing:
        #    A, monomials = self.system_matrix(psystem, log=log)

        matrix, monomials, additional_matrices = None, None, None
        if self.do_preprocessing:
            log.log("generating initial matrices")
            matrix, monomials, additional_matrices = log.timefn_log(
                self.preprocessing_base.prepare_matrices, name="prepare-matrices"
            )(psystem)
        density = log.timefn_log(self.gather_info_system)(psystem, matrix, monomials)
        # matrix=A, monomials=monomials

        if self.do_preprocessing:
            psystem, A, monomials = log.timefn_log(self.preprocess)(
                psystem, matrix, monomials, additional=additional_matrices
            )
            density = log.timefn(self.gather_info_system)(
                psystem,
                matrix=A,
                monomials=monomials,
                show_all=False,
                log=log.sublogger("gather_info_psystem-post-processing"),
            )

        sparse = self.sparse
        if sparse is None:
            log.log(
                f"Automatic sparsity decision based on matrix density; use dense if density >= {SPARSITY_TRESHOLD}"
            )
            if density < SPARSITY_TRESHOLD:
                sparse = True
                # self.enable_gpu = False
            else:
                sparse = False
        log.data("sparse", sparse)

        try:
            key_is_correct, num_solutions = log.timefn_log(
                self.compute_groebner_and_check_key
            )(psystem, key=key, sparse=sparse, method=self.gb_method)
        except MagmaGroebnerBasisInvalid:
            # print("input system")
            # niceprint_mat(A)
            # print(psystem)
            raise

        return key_is_correct

    def gather_info_system(
        self,
        psystem: PolynomialSequence_generic,
        matrix: Matrix | np.ndarray | None = None,
        monomials: tuple | None = None,
        show_all: bool = True,
        log: Logger = nolog,
    ) -> float:

        if self.is_telemetry_enabled and monomials is None:
            monomials = get_all_monomials_fast(psystem)

        system_vars = log.timefn(psystem.variables)()
        missing_vars = len(set(psystem.ring().gens()).difference(system_vars))
        log.data("missing-system-variables", missing_vars, show=True)
        log.data("system-parts", psystem.nparts(), show=show_all)
        log.data("k", sum(len(part) for part in psystem.parts()), show=True)
        try:
            log.data(
                "semireg-degree",
                log.timefn(degree_of_semiregularity_gf2)(psystem),
                show=True,
            )
        except ValueError:
            # not overdefined system
            pass

        monomials = (
            monomials if monomials is not None else get_all_monomials_fast(psystem)
        )
        len_monoms = len(monomials)
        log.data("len-monomials", len_monoms, show=True)
        log.data("ratio-monomials", len_monoms / 2**self.key_bits, show=True)
        log.data(
            "diffusion-monomials",
            np.log2(len_monoms) / self.key_bits,
            show=show_all,
        )
        log.data(
            "monomial-degree-set",
            tuple(sorted(set(monom.deg() for monom in monomials))),
            show=True,
        )

        if matrix is not None:
            log.data("matrix-dimensions", dimensions(matrix), show=show_all)
            if isinstance(matrix, np.ndarray):
                import galois

                density = (
                    log.timefn(np.sum, name="density")(matrix)
                    / matrix.shape[1]
                    / matrix.shape[0]
                )
                time0 = time()
                rank = np.linalg.matrix_rank(
                    matrix.astype(np.byte, copy=False).view(galois.GF2)
                )
                log.time("rank", time() - time0)
            else:
                density = log.timefn(matrix.density)()
                rank = matrix.rank()

            log.data("matrix-density", float(density), show=True)
            log.data("matrix-rank", int(rank), show=True)
            if rank < np.min(dimensions(matrix)):
                log.log("THERE ARE LINEARLY DEPENDENT POLYNOMIALS")
            return density
        else:
            density = log.timefn(get_density_fast)(psystem)
            log.data("matrix-density", density, show=True)
            return density

    # def system_matrix(self, psystem, log: Logger = nolog):
    #    A, monomials = log.timefn_log(fast_coefficients_monomials)(
    #        psystem,
    #        sparse=False,
    #        order="degrevlex",
    #    )
    #    return A, monomials

    def generate_aes(
        self, log: Logger = nolog
    ) -> t.Tuple[PolynomialSequence_gf2, list]:
        """Generate AES polynomial system without state variables"""

        # runs in parallel
        # SR(3,2,2,4), PC 4 ~ 15 s, SEMIREG 59
        # SR(3,2,2,4), PC 12 ~ 27 s, SEMIDEG 45

        # DETERMINE SPARSITY MANUALLY, useful for elimination
        # FOR DIFFUSED CIPHERS, SET TO FALSE
        # sparse = False

        # cannot have more samples than count of unique plaintexts

        # sim -- maybe plaintext differing by one bit?
        # meth="fgb"
        n, r, c, e = self.aes_nrce
        # use efficient representation for our polynomials when possible
        # FGb does not support this representation
        # use_pbori = self.gb_method != "fgb"
        # if use_pbori:
        #    log.log("Using efficient polybori representation of polynomials")
        # else:
        #    log.log("Using standard less efficient GF(2) representation")

        if True:
            aes = AES(n=n, r=r, c=c, e=e, polybori=True)
            adv_psystem = aes.ps_key_vars(n=self.pc_pairs, sim=False, meth="subs")
            key = np.array(list(adv_psystem.org_key)).flatten()

            psystem = PolynomialSequence(adv_psystem.polys)
            # print_time(aes.time_log(), "ps_key_vars")
            # print("Ordering:", psystem.ring().term_order())
            log.time("eliminate-state-vars", aes.t_log["ps_key_vars"])
        else:
            from ..generate import AESGenerator

            aes = AESGenerator(n=n, r=r, c=c, e=e)  # , polybori=True)
            adv_psystem, key = log.timefn(aes.ps_key_vars)(n=self.pc_pairs, meth="subs")
            key = np.array(list(adv_psystem.org_key)).flatten()

            psystem = adv_psystem
            # print_time(aes.time_log(), "ps_key_vars")
            # print("Ordering:", psystem.ring().term_order())

        return psystem, list(key)

    def preprocess(
        self, psystem, matrix=None, monomials=None, additional=None, log: Logger = nolog
    ):
        if not self.do_preprocessing:
            log.log("No preprocessing applied")
            return psystem, None, None

        pre = self.preprocessing_base
        assert pre is not None

        log.log(f"Preprocessing system with {self.preprocessing_info}")

        psystem, matrix, monomials = log.timefn(pre.run, name="pipe")(
            psystem, matrix, monomials, additional=additional, log=log
        )
        # log.timefn(get_all_monomials_fast)(psystem)
        return psystem, matrix, monomials

    def compute_groebner_and_check_key(
        self,
        psystem,
        key: list,
        sparse: bool,
        method: str = "magma",
        log: Logger = nolog,
    ) -> t.Tuple[bool, int]:
        """
        Supply key in order to check validity of result
        Returns whether key is contained in solution set and number of all solutions to the system

        Raises MagmaInvalidOutput if computation restarted too many times.
        """

        key_is_contained = False
        num_solutions = 0
        if method == "magma":
            log.log(
                f"Using {'sparse' if sparse else ('dense+GPU' if self.enable_gpu else 'dense')}"
                f" with threads={self.num_threads} Magma F4"
            )
            pack_monomials = None
            restart_count = 0
            restart_needed = False
            solution, gb_ideal = None, None
            last_exc = None

            while restart_count == 0 or restart_needed:
                restart_needed = False
                log.data("magma-restart-count", restart_count)
                if restart_count >= 3:
                    raise last_exc
                elif restart_count > 0:
                    log.log(
                        f"=========Restarting, magma run failed. Reason={last_exc}. Retry {restart_count} ====="
                    )
                restart_count += 1

                try:
                    (solution, gb_ideal) = log.timefn_log(magma_gb_solve)(
                        Ideal(psystem),
                        magma=self.magma,
                        key=self.key,
                        sparse=sparse,
                        enable_gpu=self.enable_gpu,
                        num_threads=self.num_threads,
                        pack=pack_monomials,
                    )
                except MagmaGroebnerBasisCrash as e:
                    if self.enable_gpu:
                        self.enable_gpu = False
                        # self.num_threads = MAX_CORES
                        restart_needed = True
                        last_exc = MagmaGroebnerBasisCrash(
                            "Probably crashed due to GPU usage"
                        )
                        continue
                    else:
                        restart_needed = True
                        print("Unknown Magma error")
                        last_exc = MagmaGroebnerBasisCrash(str(e))
                        continue
                    # raise

                if solution == []:
                    log.log("NO SOLUTIONS, something is not right")
                    key_is_in_original = is_key_in_ideal(Ideal(psystem), key)
                    log.log(f"original system contains key? {key_is_in_original}")
                    if key_is_in_original:
                        last_exc = MagmaGroebnerBasisInvalid(
                            "Output from magma is I=(1), ie. no solutions. But original system has solutions. Disable packing..."
                        )
                        log.log("----Disabling packing!----")
                        pack_monomials = False
                        restart_needed = True
                        continue

            if solution is not None:
                log.data("dim-variety", 1, show=True)
                log.data("dim-variety-log2", 0, show=True)
                if key == solution:
                    log.log("Unique valid key found")
                    key_is_contained, num_solutions = True, 1
                else:
                    log.log("INVALID KEY FOUND!")
                    print("found:", solution)
                    print("key:  ", key)
                    print("len(key)", len(key))
            elif gb_ideal is not None:
                if log.timefn(is_key_in_ideal)(gb_ideal, key):
                    log.log("Key is one of solutions")
                    key_is_contained = True
                else:
                    log.log("KEY IS NOT IN SOLUTION SET!")

                log.log("Determining number of solutions via Hilbert series")
                num_solutions, exact = log.timefn_log(get_num_solutions_from_gb)(
                    gb_ideal.gens()
                )
                log.data("dim-variety-is-lowerbound", not exact)
                log.data(
                    "dim-variety",
                    int(num_solutions) if num_solutions is not None else None,
                    show=True,
                )
                log.data(
                    "dim-variety-log2",
                    (
                        float(RDF(log_sage(num_solutions, 2)))
                        if num_solutions is not None
                        else None
                    ),
                    show=True,
                )

        else:  # method == "fgb":
            log.log(f"Using FGb")

            num_solutions, (_, solutions) = log.timefn_log(fgb_gb_solve)(
                Ideal(psystem), threads=24, verbosity=1
            )

        return key_is_contained, num_solutions
