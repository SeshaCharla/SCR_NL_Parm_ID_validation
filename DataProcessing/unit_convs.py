import numpy as np

"""The file contains the functions that do the appropriate unit conversions for the states and inputs from the experimental data"""
""" Standard Units:
    Temperature: deference from 250 deg-C
    Mass flow rate: g/s
    Concentration: g/ml or g/cm^3
    Area: cm^2

    Concentration   & $mol/cm^{3} = mol/ml$ 
    Time            & $s$ 
    Mass            & $g$ 
    Length          & $cm$ 
    Temperature     & $(T-250)\lx{^o}{C}$ 
"""

# Constants
kgmin2gsec = 16.6667              # Conversion factor from kg/min to g/sec
T0 = 200                               # Reference temperature in deg-C
M_nox = 30.0061                        # Molecular weight of NOx in g/mol
M_nh3 = 17.0305                        # Molecular weight of NH3 in g/mol

def uConv(x, conv_type: str):
    """Unit conversion for the states"""
    match conv_type:
        case "-T0C":
            return np.array([xi - T0 for xi in x])
        case "kg/min to g/s":
            return np.array([xi * kgmin2gsec for xi in x])
        case "NOx ppm to mol/m^3":
            return np.array([xi/M_nox for xi in x])
        case "NH3 ppm to mol/m^3":
            return np.array([xi/M_nh3 for xi in x])
        case _:
            return x
#===

