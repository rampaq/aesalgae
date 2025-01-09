import random
from time import time

import matplotlib.pyplot as plt
from aesalgae.helpers.helpers import dimensions, niceprint_mat
from aesalgae.helpers.monomials import fast_coefficients_monomials
from sage.all import Matrix, order, parallel, prod
from sage.rings.polynomial.multi_polynomial_sequence import (
    PolynomialSequence, PolynomialSequence_generic, PolynomialSequence_gf2)

from ..config import MAX_CORES
from ..Experiment import Logger, nolog
from .generic import Preprocessing


class InflateSystem(Preprocessing):
    name = "InflateSystem"
    matrix_backend = None

    def __init__(self, order_polys=False, shuffle=False, inverse=False, custom=False):
        self.order_polys = order_polys
        self.shuffle = shuffle
        self.inverse = inverse
        self.custom = custom

    def inflation_factor(self, key_bits):
        return 1 + key_bits

    def run(
        self,
        psystem: PolynomialSequence_gf2,
        matrix=None,
        monomials=None,
        additional=None,
        log: Logger = nolog,
    ):
        """Multiply each poly in system by monomials x1, ..., xn and add those to the system"""

        rng = psystem.ring()
        # inflation_factor = self.inflation_factor(rng.ngens())
        gens = psystem.ring().gens()
        inv = self.inverse
        if self.inverse:
            print("INVERSE INFLATE")
            n = psystem.ring().ngens()
            all_mon = prod(psystem.ring().gens()).lm()
            mon = [all_mon] + [
                all_mon.set().divide(gens[i].lm()).vars() for i in range(n)
            ]
            # mon = [prod(psystem.ring().gens()[:10])]
            # print(mon)
            # mon = mon[2:17]
        elif self.custom:
            print("CUSTOM")
            mon = [gens[0]]  # * gens[1] * gens[2]]
        else:
            mon = [rng.one()] + list(psystem.ring().gens())
            # mon += [mon[1] * xi for xi in psystem.ring().gens()[1:]]

        @parallel(ncpus=MAX_CORES)
        def mult_psystem_by_monom(i):
            # if not inv:
            #    if i == 0:
            #        return psystem
            mi = mon[i]
            return [mi * poly for poly in psystem]

        time0 = time()
        out = list(mult_psystem_by_monom(list(range(len(mon)))))
        if self.order_polys and not self.shuffle:
            log.timefn(out.sort)()

        if self.shuffle:
            log.timefn(random.shuffle)(out)

        polys = [system[1] for system in out]
        psystem = PolynomialSequence(polys)
        log.time("parallel constructed multiplied polys", time() - time0)

        matrix, monomials = None, None
        return psystem, matrix, monomials
