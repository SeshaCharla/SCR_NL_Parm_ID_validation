import numpy as np

"""The file contains the functions that do the appropriate unit conversions for the states and inputs from the experimental data"""
""" Standard Units:
    Temperature     : deference from 250 deg-C
    Mass flow rate  : g/s
    Concentration   : x 10^-3 mol/m^3
    urea_inj        : 10^-3 ml/sec
"""

# Constants
kgmin2gsec = 16.6667              # Conversion factor from kg/min to g/sec
T0 = 200                               # Reference temperature in deg-C
M_nox = 30.0061                        # Molecular weight of NOx in g/mol
M_nh3 = 17.0305                        # Molecular weight of NH3 in g/mol

def uConv(x, Tscr, conv_type: str):
    """Unit conversion for the states"""
    match conv_type:
        case "-T0C":
            # print("Setting reference temperature as x 10 + 200 deg-C")
            return np.array([(xi - T0)/10 for xi in x])
        case "kg/min to g/s":
            # print("Converting kg/min to g/s")
            return np.array([xi * kgmin2gsec for xi in x])
        case "ppm to 10^-3 mol/m^3":
            # print("converting ppm (mole fraction) to 10^{-3} mol/m^3")
            return np.array([(xi/(22.4*((273.15+T_scr)/(273.15)))) for (xi, T_scr) in zip(x, Tscr)])
        case "ml/s to 10 ml/s":
            # print("Scaling urea-injection to 10^{-3} ml/s")
            return np.array([xi*10 for xi in x])
        case _: # Default
            raise ValueError("Unknown unit conversion")
#===

std_units = {"T": r'$\quad [\times 10 + 200 \: ^0C]$',
             "F": r'$\quad [kg/min]$',
           "con": r'$\quad [\times 10^{-3} \, mol/m^3]$',
          "uinj": r'$\quad [\times 10 \, ml/sec]$'}

units = {"T":std_units["T"],
         "F":std_units["F"],
         "x1":std_units["con"],
         "x2":std_units["con"],
         "u1":std_units["con"],
         "eta":std_units["con"],
         "y1":std_units["con"],
         "u2":std_units["uinj"]}

