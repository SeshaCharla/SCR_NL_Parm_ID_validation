import numpy as np
import theta_sat as ths
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh
from temperature import phiT


class sat_sim:
    """ Class simulating the saturated system """
    def __init__(self, dec_dat: dd.decimatedTestData):
        """ Loads the data and generates the simulation of the saturated system """
        self.dat = dec_dat
        self.theta_sat = ths.theta_sat(self.dat)
    # ===============================================================================

    def phi_k(self):
        """ Calculates the phi(k) for getting the eta(k+1) """
