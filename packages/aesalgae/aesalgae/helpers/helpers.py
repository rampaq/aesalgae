#!/usr/bin/env python
import itertools
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sage.structure.element import Matrix as Matrix_t
from scipy.special import binom

# from sage.arith.all import binomial as binom # arbitrary precision, no numpy support

def dimensions(matrix: np.ndarray | Matrix_t):
    try:
        k, m = matrix.dimensions()
    except AttributeError:
        k, m = matrix.shape
    return k, m


def batched_it(iterable, n):
    """Batch iterator data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


### Gilbert-Varshamov
def H_entropy(x):
    """Binary entropy"""
    return np.where(
        x * (x - 1) == 0, np.zeros_like(x), -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    )


def dgv_approx(m: int, k: int):
    """Gilbert-Varsahamov distance, expected minimal distance of random [m,k] code.
    Approximated via second order Taylor and solved.
    """
    assert m >= k

    if k / m > 1 / 2:
        approx = (
            m / 2 * (1 - np.sqrt(k / m))
        )  # parabola interescting at x=0,1 -- beter global behaviour
    else:
        approx = m * (1 / 2 - np.sqrt(k / m * np.log(2) / 2))

    assert approx >= 0, "invalid d_GV approximation"
    return approx


def dgv_exact(m: int, k: int):
    """Gilbert-Varsahamov distance, expected minimal distance of random [m,k] code."""
    assert m >= k
    d0 = dgv_approx(m, k) / m

    x = sp.optimize.fsolve(lambda x: H_entropy(x) - 1 + k / m, [d0])[0]
    if x > 1 / 2:
        x = 1 - x
    return x * m


def expected_words_d_exact(m: int, k: int, d: int):
    """Expected number of codewords of give distance d"""
    return 2 ** (k - m) * binom(m, d)


def expected_words_d_bounds(m: int, k: int, d: int):
    """Get bounds on expected number of codewords with given length, lower and upper bound, respectively.
    For k < m, for k=m
    """
    assert np.all(d <= m)
    a = d / m
    h = H_entropy(a)
    comm = 2 ** (k - m + h * m) / np.sqrt(m * a * (1 - a))
    return [comm / np.sqrt(8), comm / np.sqrt(2 * np.pi)]


def plt_typical_d_curve(m, k):
    """
    Plot expected number of codewords with Hamming distance d for random code C
    Plot also Gilbert-Varshamov distance as a heuristic for minimal distance of such codes
    """

    step = np.ceil(m / 50)
    dd = np.arange(0, m + step, step)
    dd_fl = np.linspace(0.2, m - 0.2, 100)

    dgv = dgv_exact(m, k)
    bound = expected_words_d_bounds(m, k, dd_fl)

    # try to compute exact expectation value, abort on oveflow (too big numbers)
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            exact = expected_words_d_exact(m, k, dd)
        except RuntimeWarning:
            exact = None

    fig, ax = plt.subplots()

    xmin = 0
    xmax = m
    ax.set_xlim((xmin, xmax))
    ax.set_title(f"Random linear [{m},{k}]-code")
    ax.set_ylabel("Expected number of words with length $d$")
    ax.set_xlabel(r"$d$")

    ax.semilogy(dd_fl, bound[1], label="E[] upper bound")
    ax.semilogy(dd_fl, bound[0], label="E[] lower bound")
    if exact is not None:
        ax.semilogy(dd, exact, "k.", label="E[] exact")  # color="k")

    # info about d_gv and corresponding expect values
    # dgv0 = int(np.floor(dgv))
    # dgv1 = int(np.ceil(dgv))
    # dgv2 = dgv1+1
    # expected_words_fn = expected_words_d_exact if exact is not None else lambda *args: expected_words_d_bounds(*args)[1] # approximate by upper bound
    # print("d_GV:", dgv, "; expected # of vectors at d=d_GV:", expected_words_fn(m,k,dgv))
    # if dgv0 > 0:
    #    print(f"expected for d={dgv0}:", expected_words_fn(m,k,dgv0))
    # print(f"expected for d={dgv1}:", expected_words_fn(m,k,dgv1))
    # print(f"expected for d={dgv2}:", expected_words_fn(m,k,dgv2))

    ax.axvline(
        x=dgv, color="k", linestyle="--", label=r"$d_\text{GV}\approx$" + f"{dgv:.2f}"
    )  # + "\n" + r"$E[d_\text{GV}]\approx$" + f"{expected_words_fn(m,k,dgv):.2f}")

    # add x tick for gilbert varshamov dist
    # xs = np.arange(xmin,xmax,1)
    xs = ax.get_xticks()
    # clip
    # xs = xs[xmin <= xs]
    # xs = xs[xs <= xmax]
    i_dgv = np.searchsorted(xs, dgv)
    xs = np.insert(xs, i_dgv, dgv)

    # xlabels = [r"$d_\text{GV}$" if i==i_dgv else str(x) for i,x in enumerate(xs)] # " + f"{x:.2f}$"
    # xlabels = [f"{dgv:.1f}" if i==i_dgv else str(int(x)) for i,x in enumerate(xs)] # " + f"{x:.2f}$"
    xlabels = [
        "" if i == i_dgv else str(int(x)) for i, x in enumerate(xs)
    ]  # " + f"{x:.2f}$"
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    # ax.text(dgv,0,r"$d_\text{GV}$",
    #       rotation=0)

    # ax.tick_params(axis='x', labelrotation=-45)
    ax.legend(loc="upper right")
    plt.show()


### Degree elimination


def p_rnd_matrix_fullrank(k, l):
    """Probability that a given F_2^{l x k} matrix is full rank, ie rank=min(k,l)
    https://arxiv.org/pdf/1404.3250, Lemma 1
    """
    if k <= l:
        rng = range(l - k + 1, l + 1)
    else:
        rng = range(k - l + 1, k + 1)

    p = 1
    for i in rng:
        p *= 1 - 1 / 2**i
    return p


def plot_p_rnd_fullrank():
    """
    Let V_t be the subcode which has first t coordinates set to zero.
    Let l be such that V_l != {0} and V_{l+1} = {0}.

    # ints
    This is is given as points (l+1,k) such that random (l+1)xk matrix is full rank and thus V_{l+1}=0.

    new:
    probbability that random M in F_2^{l x k} has rank(M)=min(l,k)
    """
    k_lim = 50
    kk = np.arange(1, k_lim, 1)
    ll = np.arange(1, k_lim, 1)
    kks, lls = np.meshgrid(kk, ll)

    fig, ax = plt.subplots()  # subplot_kw={"projection": "3d"})

    pp = np.zeros((len(ll), len(kk)))

    # print("k", kks.shape)
    # print("l", lls.shape)
    # print("p", pp.shape)

    for ik, k in enumerate(kk):
        for il, l in enumerate(ll):
            # if l < k:
            #    continue # skip, set 0
            pp[il, ik] = p_rnd_matrix_fullrank(k, l)

    ax.plot(kks, kks, zorder=100, label="AAA")
    c = ax.pcolor(
        kks,
        lls,
        pp,
        cmap=matplotlib.cm.magma,
        linewidth=0,
        antialiased=False,
        # norm=matplotlib.colors.LogNorm(vmin=pp.min(), vmax=1)
    )
    fig.colorbar(
        c,
        ax=ax,
        label=r"$P\left[\text{random } M \in \mathbb{F}_2^{\ \ell\times k} \text{ is full rank}\right]$",
    )

    # treshidx = np.zeros_like(kk)
    # for ik, k in enumerate(kk):
    #    amin = np.argmax(pp[:,ik] > 0.9)
    #    treshidx[ik] = ll[amin]
    #
    # print("lowest values of l such that Prob > 0.9:", treshidx)
    # plt.step(kk, treshidx,"k", label=r"$P\left[\text{random } M \in \mathbb{F}_2^{\ \ell\times k} \text{ is full rank}\right] > 0.9$")

    # Customize the z axis.
    # ax.set_zlim(0, 1)
    ax.set_title("Maximum elimination in subcodes for random codes")
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\ell$")
    # ax.legend(loc="upper right")
    # ax.set_zlabel("prob")
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    plt.show()


def plot_p_fullrank_for_k(k):
    fig, ax = plt.subplots()  # subplot_kw={"projection": "3d"})

    ll = np.arange(int(k * 0.9), int(np.floor(1.5 * k)), 1)
    pp = np.zeros(len(ll))
    for il, l in enumerate(ll):
        pp[il] = p_rnd_matrix_fullrank(k, l)

    # plt.scatter(ll, pp,marker="1")
    plt.plot(ll, pp)
    plt.title(r"P[$M \in \mathbb{F}_2^{\ \ell\times k}$ is full rank], k=" + str(k))

    # Customize the z axis.
    # ax.set_zlim(0, 1)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("p")
    ax.grid()
    # ax.set_zlabel("prob")
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    plt.show()


def index_to_degree(n, ind):
    """
    Assuming that all monomials of GF(2)[x_1,...,x_n]/(x_1^2-x_1, ..., x_n^2-x_n) are present, convert index of monomial to degree with a graded monomial ordering.
    Indexing starts at 0, ie 0 is the index of monomial x_1*x_2*...*x_n
    Simply sum number of all monomials up to d such that sum > ind.
    Numpy-compatible function.
    """
    assert 0 <= np.min(ind) and np.max(ind) <= 2**n - 1, "index out of bounds"
    bsum_table = (
        np.cumsum([binom(n, dd) for dd in range(n + 1)]) - 1
    )  # is symmetric (n k) = (n n-k), -1 for 0 indexing
    bsum_table = np.atleast_2d(
        bsum_table
    ).T  # transpose to column vector so that comparison with vector is possible
    degs = n - np.argmax(bsum_table >= ind, axis=0)
    return degs if len(degs) > 1 else degs[0]


# #### Expected number of elimintaions
# - by rank-nullity theorem,
#   - number of eliminated columns $= \dim \ker M = k - \text{rank} \ M$,
#   - where we think of the matrix as a map $M: F_2^k \to F_2^\ell$.
# Hence, $\dim\ker M \geq k-l$ for $l\geq k$, although for some values, it is likely that the kernel is larger
# - from the above plot, one can observe that for $p\approx 1$, the matrices are very linkely to have full rank and thus $\text{rank} M = l$ for $l\geq k$
# - even if all such l that p < 0.9, the largest increase in kernel dimension we could hope for is given by those l, ie. at most ~5
#     - at p ~ 1, the rank is full and subsequently, it cannot decrease, hence defect increases by at most one at every l


def niceprint_mat(B):
    for v in B:
        niceprintv(v)
        # print("".join(["1" if x else "." for x in v]))


def niceprintv(v):
    print("".join(["1" if x else "." for x in v]))
