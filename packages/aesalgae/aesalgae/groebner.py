import operator
import typing as t
from functools import reduce
from time import time

from saes.config import time_log
from sage.all import GF, ZZ, Ideal, Integer, Polynomial, sage_eval
from sage.interfaces.expect import StdOutContext
from sage.interfaces.magma import Magma, MagmaGBLogPrettyPrinter
from sage.parallel.decorate import parallel
# import numpy as np
from sage.rings.ideal import Ideal_generic
from sage.rings.polynomial.multi_polynomial_ideal import MPolynomialIdeal
from sage.rings.polynomial.multi_polynomial_sequence import (
    PolynomialSequence, PolynomialSequence_gf2)
from sage.rings.polynomial.pbori.pbori import BooleanPolynomialIdeal

from .config import (MAX_CORES, MagmaGroebnerBasisCrash,
                     MagmaGroebnerBasisInvalid)
from .Experiment import Logger, nolog
from .helpers import batched_it
# from sage.crypto.mq import sr
from .helpers.logparser import MagmaGBLogger


def magma_ideal(
    ideal: BooleanPolynomialIdeal,
    magma: Magma,
    key,
    pack: bool | None = None,
    log: Logger = nolog,
):
    """
    For BooleanPolynomialRing.
    Magma crashes with sage interface for GB when there are too many monomials in polynomials.
    Hence, construct the polynomials one by one iteratively in chunks by using no more that fixed amount of monomials at a time
    SR(3,2,2,4) PC2 ~ 1min to convert without packing
    SR(3,2,2,4) PC2 ~ 5s to convert with packing
    SR(3,2,2,4) PC2 ~ <1s to convert with parallelized packing

    Bug with packing in Magma: for n=31,32 variables, the packing is bugged; temporary
    solution=artificially increase the variable count by blank variables to 33 vars.

    Use packing for larger systems (Magma):
        For SR(2,4,4,4) PC=1, magma_ideal takes ~ 300s.  Looking at htop, python is
        finished after ~7 seconds and magma then digests the input.
        This packing is ~20x faster.

        BooleanPolynomial(B, Q) : RngMPolBool, [RngIntElt] -> RngMPolBoolElt
        (in documentation as BooleanPolynomialRing(B,Q) -- wrong)

        Given a boolean polynomial ring B of rank n and a sequence Q of integers, create
        the boolean polynomial in B whose monomials are given by the entries of Q: each
        integer must be in the range [0 ... 2^n - 1] and its binary expansion gives the
        exponents of the monomial in order (the resulting monomials are sorted w.r.t.
        the monomial order of B, so may be given in any order and duplicate monomials
        are added).

        This function is simply provided so that boolean polynomials may be stored and
        read back in a compact form; otherwise, one can create a boolean polynomial in
        the usual way from the generators of B after B is created. Note also that if one
        prints B, an ideal of B, or an element of B with the Magma print level, then
        this function will be used to print the elements in a compact form.

    Args:
        ideal: sage pbori BooleanPolynomialIdeal polynomial ideal to convert to magma ideal
        magma: sage's magma interface
        pack: whether to pack BooleanPolynomials for magma - massive speed boost for converting the ideal
    """
    assert isinstance(ideal, BooleanPolynomialIdeal)

    rng = ideal.ring()
    key_bits = rng.ngens()

    # init ring
    # coerce grevlex. input ideal might be in another ordering but
    order = "grevlex"
    # order="lex"
    # order = rng.term_order().magma_str()
    # P<a,b,c,d,e> := BooleanPolynomialRing(5, "grevlex");

    variables = list(rng.gens())

    # bug in Magma when nvars=31,32 with packed representation. will be fixed in next magma release
    blank_vars = 0
    if key_bits == 31:
        blank_vars = 2
    elif key_bits == 32:
        blank_vars = 1
    variables += [f"blank{i}" for i in range(blank_vars)]

    magma_rng = "P"
    magma_rng_str = (
        f'P<{",".join(map(str, variables))}> :='
        f'BooleanPolynomialRing({len(variables)}, "{order}")'
    )
    magma.eval(magma_rng_str)
    gens = ideal.gens()

    # ideal generators
    poly_names = []
    key_bits = ideal.ring().ngens()  # no of variables
    if pack is False or pack is None and key_bits <= 8:
        # do not pack
        MAX_MONOM = 10_000
        log.log("Naively transcribing polynomials for magma")
        time0 = time()
        for i, poly in enumerate(gens):
            poly_names.append(f"p{i}")
            magma.eval(f"p{i} := 0")
            for mon_chunks in batched_it(poly, MAX_MONOM):
                chunked = "+".join(str(mon) for mon in mon_chunks)
                magma.eval(f"p{i} := p{i} + " + chunked)
        log.time("load polys to magma", time() - time0)
    else:
        # ring orderings do not matter for packing - always in lex
        # var[i] -> 2^i
        log.log("Packing polynomials for magma")
        time0 = time()
        parallelized_pack = list(sorted(pack_poly(list(enumerate(gens)))))
        log.time("constructed packed polys - parallel", time() - time0)

        time0 = time()
        for args, magma_str in parallelized_pack:
            (i, _), _ = args
            poly_names.append(f"p{i}")
            # magma.eval(f"packed{i}:=" + magma_str)
            # magma.eval(f"p{i} := BooleanPolynomial({magma_rng}, packed{i})")
            magma.eval(f"p{i} := BooleanPolynomial({magma_rng}, {magma_str})")
        log.time("transfered packed polynomials to magma", time() - time0)

    magma_gens = magma("[" + ",".join(poly_names) + "]")
    return magma("ideal<%s|%s>" % (magma_rng, magma_gens.name())), blank_vars


def magma_gb_solve(
    ideal: BooleanPolynomialIdeal,  # BooleanPolynomialIdeal,
    magma,
    sparse: bool,
    key,
    enable_gpu: bool = True,
    num_threads: int = 1,
    pack: bool | None = None,
    prot="sage",
    log: Logger = nolog,
) -> t.Tuple[list, None] | t.Tuple[None, BooleanPolynomialIdeal]:
    """
    Compute Groebner basis of BooleanPolynomial system using Magma and output corresponding variety.

    prot: True -- magma
          "sage" -- format with sage
          disable_gpu: when using dense variant, disable GPU to prevent crashes likely coming from insufficient GPU memory
          disable_threads

    Returns:
        (empty list if variety is empty, None),
        (the unique solution in F2 is variety is a single point, given as a list of {0,1}, None),
        (None, grobner basis) if the solution is not unique. Size of variety is determined from GB usign Hilbert Series.

        When you want to test if a point is contained in a variety, simply substitute to the groebner basis and look if it evaluates to 0.
    Raises:
        TypeError
    """

    bool_rng = ideal.ring()
    key_bits = ideal.ring().ngens()  # no of variables

    # usage of GPU for CUDA-enabled executable and supported algorithms (dense F4)
    log.log(
        f"Magma options: sparse={sparse}, GPU={enable_gpu}, num_threads={num_threads}"
    )
    magma.SetGPU("true" if enable_gpu else "false", nvals=0)
    magma.SetNthreads(num_threads)

    magma.ResetMaximumMemoryUsage(nvals=0)
    magma_idl, blank_vars = log.timefn_log(magma_ideal)(
        ideal,
        magma,
        key=key,
        pack=pack,
    )

    log_parser = None
    if prot:
        log_parser = MagmaGBLogger(
            verbosity=1, style="sage" if prot == "sage" else "magma", log=log
        )
        magma.SetVerbose("Groebner", 1)

    time0 = time()
    try:
        log.log("Starting GB")
        with StdOutContext(magma, silent=not prot, stdout=log_parser):
            # DynamicStrategy, Dense (automatic)
            gb = magma_idl.GroebnerBasis(
                Faugere=True,
                Boolean=True,
                Dense=not sparse,
                # DynamicStrategy=True, # doesnt halt
                nvals=1,  # v2.28 also outputs degrees encountered as a second argument (nvals=2)
            )  # -> (GB sequence, List of degrees in F4)
    except TypeError as e:
        log.time("gb-crash", time() - time0)
        if prot == "sage" and log_parser is not None:
            degrees, critical_pairs = log_parser.get_stats()
            log.data("degrees", degrees, show=False)
            log.data("critical_pairs", critical_pairs, show=False)
        raise MagmaGroebnerBasisCrash() from e
    log.time("gb", time() - time0)

    # log.data("degs", degs)
    if prot == "sage":
        assert log_parser is not None
        log.data("highest-deg", log_parser.max_deg)
        degrees, critical_pairs = log_parser.get_stats()
        log.data("degrees", degrees, show=False)
        log.data("critical_pairs", critical_pairs, show=False)

    # gb_sage = gb.sage()
    list_gb = list(gb)
    len_gb = len(list(gb))
    print(len_gb)

    mem = int(magma.GetMaximumMemoryUsage()) // 1_000_000
    log.data("memory_MB", mem)
    magma.ResetMaximumMemoryUsage(nvals=0)

    log.data("len-groebner", len_gb)
    # log.data("len(gb[0])", list_gb[0].Length(nvals=1))
    # log.data(
    #    "hilbert-dimension",
    #    PolynomialSequence(gb_sage, bool_rng).ideal().hilbert_polynomial().degree(),
    # )
    if len_gb == 1 and list(gb)[0] == ideal.ring().one():
        log.log("Magma GB", list(map(str, list_gb)))
        log.log("SYSTEM HAS NO SOLUTION!")
        return ([], None)

    elif len_gb == key_bits:
        # single solution
        print(list_gb)
        time0 = time()
        solution = gb.Ideal().Variety()  # ("Variety(Ideal(gb));")
        log.time("variety", time() - time0)
        # magma indexes starting from 1, delete the blank zero variables at the end
        solution = list(solution[1])[:key_bits]
        return (solution, None)
    else:
        # many solutions
        locals = dict(zip(bool_rng.variable_names(), bool_rng.gens()))
        gb = bool_rng.ideal([sage_eval(str(f), locals=locals) for f in gb])
        # log.data("gb", list(map(str, gb.gens())))
        # if len_gb < key_bits - 2:
        #    # do not compute variety, might take a long time, present estimate for
        #    # linear groebner basis
        #    return 2 ** (len_gb - key_bits), (False, gb)
        # else:
        return (None, gb)


def get_num_solutions_from_gb(
    gb: PolynomialSequence_gf2, log: Logger = nolog
) -> t.Tuple[Integer, bool]:
    """Compute Hilbert Series and deduce dimension of variety
    If dimension is expected to be too high to compute in reasonable time, return
        lower bound, False
    else
        variety size, True

    """
    if Ideal(gb).is_zero():
        log.log("System is zero ideal")
        return 2 ** (gb.ring().ngens())

    bool_rng = gb.ring()
    len_diff = gb.ring().ngens() - len(gb)
    # gb.ring().ngens() >= 64
    if bool_rng.ngens() >= 64 or len_diff >= 16:
        # singular is bugged for >= 64 variables
        # hilbert series computation would take too long
        if len_diff < 0:
            return None, False
        return 2**len_diff, False

    gf2_rng = bool_rng.change_ring(GF(2))
    gf2_ideal_fieldeqns = gf2_rng.ideal(
        [f.lm() for f in gb] + [xi**2 for xi in gf2_rng.gens()]
    )
    time0 = time()
    # Cox,Little define it as dim R_{<=s}/I_{<=s}, sage as R_s/I_s, so we need to add all coefficients
    hilbert_series = gf2_ideal_fieldeqns.hilbert_series()
    log.data("hilbert-series", str(hilbert_series))
    variety_size = hilbert_series(1)
    log.time("variety dimension using Hilbert series", time() - time0)

    # variety_size2 = len(
    #    log.timefn(
    #        gf2_ideal_fieldeqns.normal_basis,
    #        "compute normal basis from groebner to get #solutions",
    #    )()
    # )
    # assert variety_size == variety_size2
    return variety_size, True


def is_key_in_ideal(ideal: MPolynomialIdeal, key: list):
    """After suplying list of polynomials, return whether its variety contains given key"""
    # https://ask.sagemath.org/question/49683/the-number-of-solutions-of-a-polynomial-system-using-a-grobner-basis/
    res = ideal.subs({xi: key[i] for i, xi in enumerate(ideal.ring().gens())})
    return res.is_zero()  # zero ideal


def degree_of_semiregularity_gf2(psystem: PolynomialSequence_gf2):
    """Semiregularity degree for polynomials in K[x1,...xn]/(x1^2-x1,...,xn^2-xn).

    modified original version in multi_polynomial_ideal.degree_of_semi_regularity
    Magali Bardet, Jean-Charles Faugère, Bruno Salvy. Complexity of Gröbner basis computation for
    Semi-regular Overdetermined sequences over F_2 with solutions in F_2. [Research Report] RR-5049,
    INRIA. 2003.
    """
    ideal = Ideal(psystem)
    degs = [f.degree() for f in ideal.gens() if f != 0]  # we ignore zeroes
    m, n = ideal.ngens(), len(set(sum([f.variables() for f in ideal.gens()], ())))
    if m <= n:
        raise ValueError("This function requires an overdefined system of polynomials.")

    from sage.misc.misc_c import prod
    from sage.rings.power_series_ring import PowerSeriesRing
    from sage.rings.rational_field import QQ

    z = PowerSeriesRing(QQ, "z", default_prec=sum(degs)).gen()
    s = (1 + z) ** n / prod([1 + z**d for d in degs])
    for dreg in range(sum(degs)):
        if s[dreg] <= 0:
            return ZZ(dreg)
    raise ValueError("BUG: Could not compute the degree of semi-regularity")


def ideal_gf2(ideal: BooleanPolynomialIdeal) -> Ideal_generic:
    """
    Convert polybori BooleanPolynomialIdeal in GF(2)[x1,...,xn]/(x1^2-x1,...,xn^2-xn) to ideal in GF(2)[x1,...,xn]

    Slow af.

    The lift GF(2)[x1,...,xn]/(x1^2-x1,...,xn^2-xn) -> GF(2)[x1,...,xn] converts the
    underlying technical implementation and adds field variables x1^2 -x1, ...

    """

    polys = ideal.gens()
    new_rng = ideal.ring().change_ring(GF(2), order="degrevlex")
    new_vars = new_rng.gens()

    # converted_polys = list(
    #    sum(  # add new monomials
    #        map(
    #            lambda old_monom: reduce(
    #                lambda new_mon, idx: new_mon * new_vars[idx],
    #                old_monom.iterindex(),
    #                new_rng.one(),  # reduce: [0, 1, 3] --> x0*x1*x3
    #            ),
    #            poly.set(),
    #        )
    #    )
    #    for poly in gens
    # )  # x0*x1*x3 + x0*x1 -> [[0,1,3], [0,1]] -> [ 2^0 + 2^1 + 2^2, 2^0 + 2^1 ] = [ 7, 3 ]

    @parallel(ncpus=MAX_CORES)
    def convert_poly(i):
        poly = polys[i]
        new_poly = new_rng.zero()
        for old_monom in poly.set():
            new_monom = new_rng.one()
            monom_idxs = old_monom.iterindex()
            for idx in monom_idxs:
                new_monom *= new_vars[idx]
            new_poly += new_monom
        return new_poly

    converted_polys = [poly[1] for poly in convert_poly(list(range(len(polys))))]
    field_eqns = []  # xi**2 - xi for xi in new_vars]

    return new_rng.ideal(converted_polys + field_eqns)


def fgb_gb_solve(
    ideal: BooleanPolynomialIdeal, threads: int, verbosity: int = 0, log: Logger = nolog
):
    import fgb_sage

    # lift into GF(2)[x_1,...,x_n] from GF(2)[x_1,...,x_n]/(x1^n-x1,..).
    # have to test whether field equations are included in the new ring
    # probably not
    # ideal = ideal.change_ring(new_rng)

    ideal = log.timefn(ideal_gf2)(ideal)
    log.log("Converted system for singular/fgb")
    new_vars = ideal.ring().gens()
    gb = log.timefn(fgb_sage.groebner_basis)(
        ideal,
        threads=threads,
        verbosity=verbosity,
        matrix_bound=2**27,
        max_base=2**27,
    )
    print()
    variety = gb.ideal().variety()
    solutions = [[varietypoint[var] for var in new_vars] for varietypoint in variety]
    return len(solutions), (True, solutions)


@parallel(ncpus=MAX_CORES)
def pack_poly(i, poly):
    poly_packed_ints = map(
        lambda monom: reduce(lambda x, y: x | Integer(1 << y), monom.iterindex(), 0),
        poly.set(),
    )  # x0*x1*x3 + x0*x1 -> [[0,1,3], [0,1]] -> [ 2^0 + 2^1 + 2^2, 2^0 + 2^1 ] = [ 7, 3 ]
    return "[" + ",".join(map(str, poly_packed_ints)) + "]"
