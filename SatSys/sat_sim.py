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
        self.eta_sim = self.sim_eta()
        self.data_len = self.theta_sat.cAb.data_len
    # ===============================================================================

    def phi_sat(self, k) -> np.ndarray:
        """ Calculates the phi(k) for getting the eta(k+1) """
        u1_k = self.dat.ssd['u1'][k]
        F_k = self.dat.ssd['F'][k]
        T_k = self.dat.ssd['T'][k]
        phi_k = phiT.phi_T(T_k)
        return (u1_k/F_k)*phi_k
    # ============================================================

    def sim_eta(self):
        """ Simulate the eta from data """
        eta_sim = np.zeros(self.data_len)
        for k in range(self.data_len):


