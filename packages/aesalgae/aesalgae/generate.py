""" System generation, based on work of Marek Bielik """
import numpy as np
from sage.all import GF
from sage.crypto.mq import sr
from sage.misc.flatten import flatten as sage_flatten
from sage.parallel.decorate import parallel
from sage.rings.polynomial.multi_polynomial_sequence import (
    PolynomialSequence, PolynomialSequence_gf2)
from sage.rings.polynomial.pbori import pbori
from sage.rings.polynomial.pbori.pbori import (BooleanPolynomialIdeal,
                                               BooleanPolynomialRing)
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

from .config import MAX_CORES
from .Experiment import Logger, nolog

ZERO_INVERSION_LIMIT = 500


def parallelize(fn, arg_list):
    return [item[1] for item in fn(arg_list)]


class AESGenerator:

    def __init__(self, n, r, c, e, key=None):
        self.aes_tup = (n, r, c, e)
        self.key = key
        self.aes = sr.SR_gf2(n, r, c, e, gf2=True, polybori=True)
        self.key_bits = r * c * e

    def gen_systems(self, systems: int = 1):
        """Returns list of systems corresponding to a single key and multiple plaintexts

        Each system is comprised of a list of structures in aes. It comes as a list of
        size n rounds with each element cosisting of pairs, first is subsystem for key
        schedule, sboxes etc second ."""

        key_np = np.random.randint(0, 2, size=self.key_bits, dtype="bool")
        key_aes = self.aes.vector(list(key_np))
        plaintexts = []
        psystems = []

        zero_inversions = 0
        for _ in range(systems):
            while zero_inversions <= ZERO_INVERSION_LIMIT:
                plaintext = list(
                    np.random.randint(0, 2, size=self.key_bits, dtype="int")
                )
                if plaintext in plaintexts:
                    # this plaintext was already generated, restart
                    continue
                plaintext_aes = self.aes.vector(plaintext)
                try:
                    # when zero inversion occurs, try different plaintext
                    ciphertext_aes = self.aes(P=plaintext_aes, K=key_aes)
                    F, _ = self.aes.polynomial_system(
                        P=plaintext_aes, C=ciphertext_aes
                    )  # P=plaintext, K=key)
                    break
                except ZeroDivisionError:
                    zero_inversions += 1
                    continue
            else:
                raise ValueError(
                    f"Too many 0-inversions occured (> {ZERO_INVERSION_LIMIT})"
                )
            plaintexts.append(plaintext)
            psystems.append(F)

        return psystems, key_aes

    def run_aux(self, flatten: bool = True, log: Logger = nolog) -> tuple:
        """Flatten is used to erase any information about structure of AES, ie. rounds and sboxes etc"""
        psystems, key = log.timefn(self.gen_systems)(systems=1)
        psystem = (psystems[0],)
        psystem = sage_flatten(psystem, max_level=2) if flatten else psystem
        return PolynomialSequence(psystem), key

    def new_ps_key(self):
        def sboxes_gb(sboxes):
            return (
                BooleanPolynomialIdeal(
                    BooleanPolynomialRing(
                        names=sbox[0].variables(),
                        order=f"deglex({W_NUM}), deglex({W_NUM})",
                    ),
                    sbox,
                ).groebner_basis()[:W_NUM]
                for sbox in (
                    sboxes[i : i + SBOX_OFFSET]
                    for i in range(0, len(sboxes), SBOX_OFFSET)
                )
            )

        def sep_lm(polys):
            return {f"{p.lm()}": p.subs({p.lm(): 0}) for p in polys}

        def sub_vars(polys, vars_):
            @parallel(ncpus=NCPUS)
            def substitute_variables(poly):
                return pbori.substitute_variables(ring, gens, poly)

            if not polys:
                return []
            else:
                gens = [
                    ring(vars_[f"{i}"]) if f"{i}" in vars_ else ring(i)
                    for i in polys[0].ring().gens()
                ]

                return _par(substitute_variables, list(polys))

        @parallel(ncpus=NCPUS)
        def elim_aux_vars(ps):
            @parallel(ncpus=NCPUS)
            def trim_poly(poly):
                return ring.remove_var(*elim_vars)(poly)

            w_vars = {}
            s_vars = {}
            elim_vars = []
            ps_raw_it = iter(ps.parts())
            for r_polys, ks_polys in zip(ps_raw_it, ps_raw_it):
                vars_to_sub = {}
                x_vars = {}

                # Express the key schedule in the vars of the initial key.
                for sbox in sboxes_gb(ks_polys[self.s_size :]):
                    x_vars.update(sep_lm(sbox))

                elim_vars += list(x_vars.keys())
                s_vars = sep_lm(
                    sub_vars(sub_vars(ks_polys[: self.s_size], x_vars), s_vars)
                )

                vars_to_sub.update(s_vars)

                # Express the round in the vars of the initial key.
                for sbox in sboxes_gb(r_polys[self.s_size :]):
                    vars_to_sub.update(sep_lm(sub_vars(sbox, w_vars)))

                elim_vars += list(w_vars.keys()) + list(vars_to_sub.keys())
                polys = sub_vars(r_polys[: self.s_size], vars_to_sub)
                w_vars = sep_lm(polys)

            return _par(trim_poly, polys)

        @parallel(ncpus=NCPUS)
        def anf(pt, ct):
            @parallel(ncpus=NCPUS)
            def poly_anf(i):
                def ct_vec(i):
                    with sr.AllowZeroInversionsContext(self):
                        return [
                            c[i][0]
                            for c in (
                                self(P=pt, K=self.vector(k))
                                for k in itertools.product((0, 1), repeat=self.s_size)
                            )
                        ]

                return BoolFun(ct_vec(i)).algebraic_normal_form() + ct[i][0]

            return _par(poly_anf, list(range(self.s_size)))

        def fgb(poly_systems):
            import fgb_sage

            nonlocal ring
            R = PolynomialRing(
                GF(2), len(ring.variable_names()), ", ".join(ring.variable_names())
            )
            ring = ring.remove_var(*ring.gens()[: -self.s_size])
            polys = [
                flatten(
                    fgb_sage.eliminate(
                        [R(p) for p in flatten(ps.parts())],
                        R.gens()[: -self.s_size],
                        matrix_bound=2**27,
                        max_base=2**27,
                        verbosity=0,
                        threads=NCPUS,
                    ).parts()
                )
                for ps in poly_systems
            ]

            @parallel(ncpus=NCPUS)
            def adjust_polys(polys):
                @parallel(ncpus=NCPUS)
                def adjust_poly(poly):
                    return ring(poly)

                return _par(adjust_poly, polys)

            return _par(adjust_polys, polys)

        ps = self._ps_raw(n, sim, key)
        ring = ps["ring"]

        if self.e == 8:
            W_NUM = 8
            SBOX_OFFSET = 24
        else:
            W_NUM = 4
            SBOX_OFFSET = 12

        if meth == "subs":
            ps["polys"] = _par(elim_aux_vars, ps["ps"])
        elif meth == "ANF":
            ps["polys"] = _par(anf, list(zip(ps["pt"], ps["ct"])))
        elif meth == "fgb":
            ps["polys"] = fgb(ps["ps"])
        else:
            raise ValueError("Unknown meth.")

        ring = ps["polys"][0][0].ring()
        ps["ring"] = ring
        ps["vars"] = ring.gens()

        return PolynomialSystem(**ps)

    def ps_key_vars(self, n=1, meth="subs"):
        psystems, key = self.gen_systems(systems=n)
        ring = psystems[0][0].ring()

        _, _, _, e = self.aes_tup
        if e == 8:
            W_NUM = 8
            SBOX_OFFSET = 24
        else:
            W_NUM = 4
            SBOX_OFFSET = 12

        def sboxes_gb(sboxes):
            return (
                BooleanPolynomialIdeal(
                    BooleanPolynomialRing(
                        names=sbox[0].variables(),
                        order=f"deglex({W_NUM}), deglex({W_NUM})",
                    ),
                    sbox,
                ).groebner_basis()[:W_NUM]
                for sbox in (
                    sboxes[i : i + SBOX_OFFSET]
                    for i in range(0, len(sboxes), SBOX_OFFSET)
                )
            )

        def sep_lm(polys):
            return {f"{p.lm()}": p.subs({p.lm(): 0}) for p in polys}

        def sub_vars(polys, vars_):
            @parallel(ncpus=NCPUS)
            def substitute_variables(poly):
                return pbori.substitute_variables(ring, gens, poly)

            if not polys:
                return []
            else:
                gens = [
                    ring(vars_[f"{i}"]) if f"{i}" in vars_ else ring(i)
                    for i in polys[0].ring().gens()
                ]

                return _par(substitute_variables, list(polys))

        @parallel(ncpus=MAX_CORES)
        def elim_aux_vars(ps):

            w_vars = {}
            s_vars = {}
            elim_vars = []
            ps_raw_it = iter(ps.parts())
            for r_polys, ks_polys in zip(ps_raw_it, ps_raw_it):
                vars_to_sub = {}
                x_vars = {}

                # Express the key schedule in the vars of the initial key.
                print("AAAAAA", ks_polys)

                for sbox in sboxes_gb(ks_polys[self.key_bits :]):
                    x_vars.update(sep_lm(sbox))
                print("XXXX", x_vars)

                elim_vars += list(x_vars.keys())
                s_vars = sep_lm(
                    sub_vars(sub_vars(ks_polys[: self.key_bits], x_vars), s_vars)
                )

                vars_to_sub.update(s_vars)

                # Express the round in the vars of the initial key.
                for sbox in sboxes_gb(r_polys[self.key_bits :]):
                    vars_to_sub.update(sep_lm(sub_vars(sbox, w_vars)))
                    print("VARS_TO_SUB", vars_to_sub)

                elim_vars += list(w_vars.keys()) + list(vars_to_sub.keys())
                polys = sub_vars(r_polys[: self.key_bits], vars_to_sub)
                w_vars = sep_lm(polys)

            print("POLYS", polys)

            @parallel(ncpus=MAX_CORES)
            def trim_poly(poly):
                # [str(var) for var in elim_vars]
                return ring.remove_var(*elim_vars)(poly)

            return parallelize(trim_poly, polys)

        if meth == "subs":
            final_system = parallelize(elim_aux_vars, psystems)
        elif meth == "fgb":
            final_system = self.fgb_elim(ring, psystems)
        else:
            raise ValueError("Unknown meth.")

        return PolynomialSequence(final_system), key

    def fgb_elim(self, ring, psystems):
        import fgb_sage

        R = PolynomialRing(
            GF(2), len(ring.variable_names()), ", ".join(ring.variable_names())
        )
        ring = ring.remove_var(*ring.gens()[: -self.key_bits])
        polys = [
            sage_flatten(
                fgb_sage.eliminate(
                    [R(p) for p in sage_flatten(ps.parts())]
                    + [xi**2 - xi for xi in R.gens()],  # [: -self.key_bits]],
                    R.gens()[: -self.key_bits],
                    matrix_bound=2**27,
                    max_base=2**27,
                    # verbosity=0,
                    # threads=MAX_CORES,
                ).parts()
            )
            for ps in psystems
        ]

        @parallel(ncpus=MAX_CORES)
        def adjust_polys(polys):
            @parallel(ncpus=MAX_CORES)
            def adjust_poly(poly):
                return ring(poly)

            return parallelize(adjust_poly, polys)

        return parallelize(adjust_polys, polys)
