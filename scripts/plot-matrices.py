""" Plot matrices encountered in preprocessing """
import aesalgae
import matplotlib.pyplot as plt
import numpy as np
from aesalgae.helpers import fast_coefficients_monomials
from sage.all import tmp_filename

FOLDER = "matrices"
FNAME = "{aes}_{pc}_{preprocessing}{rdim}.pdf"

# a = [
#    {"aes": 2224, "pc": 1, "preproc": None},
#    {"aes": 2224, "pc": 1, "preproc": "inflate"},
#    {"aes": 2224, "pc": 2, "rdim": 1, "preproc": "monomelim"},
#    {"aes": 2224, "pc": 1, "rdim": 1, "preproc": "lll"},
#    {"aes": 2224, "pc": 2, "rdim": 1, "preproc": "monomelim,lll"},
#    #
#    {"aes": 3224, "pc": 1, "preproc": None},
#    {"aes": 3224, "pc": 1, "preproc": "inflate"},
#    {"aes": 3224, "pc": 2, "rdim": 1, "preproc": "monomelim"},
#    {"aes": 3224, "pc": 1, "rdim": 1, "preproc": "lll"},
#    {"aes": 3224, "pc": 2, "rdim": 1, "preproc": "monomelim,lll"},
#    #
#    {"aes": 2444, "pc": 1, "preproc": None},
#    {"aes": 2444, "pc": 2, "rdim": 1, "preproc": "monomelim"},
#    {"aes": 2444, "pc": 1, "rdim": 1, "preproc": "lll"},
#    {"aes": 2444, "pc": 2, "rdim": 1, "preproc": "monomelim,lll"},
#    #
#    {"aes": 1448, "pc": 1, "preproc": None},
#    {"aes": 1448, "pc": 2, "rdim": 1, "preproc": "monomelim"},
#    {"aes": 1448, "pc": 1, "rdim": 1, "preproc": "lll"},
#    {"aes": 1448, "pc": 2, "rdim": 1, "preproc": "monomelim,lll"},
#    #
#    {"aes": 2424, "pc": 1, "preproc": None},
#    {"aes": 2424, "pc": 2, "rdim": 1, "preproc": "monomelim"},
#    {"aes": 2424, "pc": 1, "rdim": 1, "preproc": "lll"},
#    {"aes": 2424, "pc": 2, "rdim": 1, "preproc": "monomelim,lll"},
# ]
# a = [
#    # {"aes": 1448, "pc": 1, "preproc": None},
#    # {"aes": 1448, "pc": 4, "preproc": None},
#    {"aes": 1448, "pc": 16, "preproc": None},
#    # {"aes": 1448, "pc": 1, "preproc": None},
#    {"aes": 1448, "pc": 4, "rdim": 2, "preproc": "monomelim"},
#    {"aes": 1448, "pc": 16, "rdim": 2, "preproc": "lll"},
#    {"aes": 1448, "pc": 4, "rdim": 2, "preproc": "monomelim,lll"},
#    {"aes": 2424, "pc": 16, "rdim": 2, "preproc": "lll"},
#    {"aes": 2424, "pc": 32, "rdim": 1, "preproc": "monomelim"},
#    {"aes": 2424, "pc": 32, "rdim": 1, "preproc": "monomelim,lll"},
# ]
a = [
    # {"aes": 1448, "pc": 1, "preproc": None},
    # {"aes": 1448, "pc": 4, "preproc": None},
    {"aes": 2424, "pc": 2, "preproc": None},
    {"aes": 2244, "pc": 1, "preproc": None},
    {"aes": 2244, "pc": 32, "preproc": None},
    # {"aes": 1448, "pc": 1, "preproc": None},
    {"aes": 2244, "pc": 8, "rdim": 1, "preproc": "monomelim"},
    {"aes": 2244, "pc": 4, "rdim": 2, "preproc": "lll"},
    {"aes": 2244, "pc": 8, "rdim": 1, "preproc": "monomelim,lll"},
    # {"aes": 2424, "pc": 16, "rdim": 2, "preproc": "lll"},
    # {"aes": 2424, "pc": 32, "rdim": 1, "preproc": "monomelim"},
    # {"aes": 2424, "pc": 32, "rdim": 1, "preproc": "monomelim,lll"},
]

for exp in a:
    aestup = exp["aes"]
    n, r, c, e = map(int, str(aestup))
    preproc = exp["preproc"]
    pc = exp["pc"]
    rdim = exp["rdim"] * r * c * e if "rdim" in exp else None

    path = (
        FOLDER + "/" + FNAME.format(aes=aestup, preprocessing=preproc, rdim=rdim, pc=pc)
    )
    print(path)
    aes = aesalgae.AlgebraicAES(
        n,
        r,
        c,
        e,
        pc_pairs=pc,
        reduced_dim=rdim,
        preprocessing=preproc,
    )
    if preproc == "inflate":
        aes.preprocessing_base = aesalgae.preprocessing.inflate.InflateSystem(
            order_polys=True
        )

    psystem, key = aes.generate_aes()
    matrix, monomials = None, None
    if preproc:
        psystem, matrix, monomials = aes.preprocessing_base.run(psystem)

    if not isinstance(matrix, np.ndarray):
        matrix, monomials = fast_coefficients_monomials(psystem, backend="numpy")

    # plt.matshow(matrix, cmap="binary")
    # plt.gca().set_aspect("auto")
    fig = plt.figure(figsize=(20, 5))
    plt.imshow(
        matrix, aspect="auto", cmap="binary", vmin=0, vmax=1, interpolation="none"
    )
    # plt.tight_layout()
    # plt.show()
    plt.savefig(path)
