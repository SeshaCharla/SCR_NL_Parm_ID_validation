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
        self.data_len = self.theta_sat.cAb.data_len
        self.eta_sim = self.sim_eta()

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
        for k in range(self.data_len-1):
            i = sh.get_interval_T(self.dat.ssd['T'][k])
            eta_sim[k+1] = ((self.phi_sat(k)).T @ self.theta_sat.thetas[sh.part_keys[i]])[0, 0]
        eta_sim[0] = eta_sim[1]
        return eta_sim


# Testing
if __name__ == "__main__":
    import pprint as pp
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    sim = sat_sim(dd.decimatedTestData(1, 1))

    plt.figure()
    plt.plot(sim.dat.ssd['t'], sim.dat.ssd['eta'], label='eta from data set')
    plt.plot(sim.dat.ssd['t'], sim.eta_sim, label='eta_saturated')
    plt.legend()
    plt.grid()
    plt.show()
