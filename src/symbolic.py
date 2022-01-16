import sympy


def getModelPointSymbols():
    return tuple(sympy.symbols("X Y Z"))


def getExtrinsicSymbols():
    return tuple(sympy.symbols("ρx ρy ρz tx ty tz"))


def getHomographySymbols():
    return tuple(sympy.symbols("H11 H12 H13 H21 H22 H23 H31 H32 H33"))
