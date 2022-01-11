import sympy


def getModelPointSymbols():
    return tuple(sympy.symbols("X Y Z"))


def getExtrinsicSymbols():
    return tuple(sympy.symbols("ρx ρy ρz tx ty tz"))

