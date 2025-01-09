MAX_CORES = 24


class MagmaGroebnerBasisCrash(Exception):
    """Magma GroebnerBasis function crashed"""


class MagmaGroebnerBasisInvalid(Exception):
    """Magma outputs Groebner basis ideal as (1), so no solutions exists. But key is
    contained in input variety - magma failed for some reason"""


class PreprocessingTooHighOutputDim(Exception):
    """Output dimension is too high and the number of linearly independent polynomials
    in input system is smaller that output dimension"""


class PreprocessingParamsInvalid(Exception):
    pass
