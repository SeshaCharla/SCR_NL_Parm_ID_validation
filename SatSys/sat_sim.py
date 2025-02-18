import numpy as np
import theta_sat as ths
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh
from temperature import phiT


class sat_sim:
    """ Class simulating the saturated system """
    def __init__(self, dec_dat: dd.decimatedTestData, T_ord: int , T_parts: list ):
        """ Loads the data and generates the simulation of the saturated system """
        self.dat = dec_dat
        self.T_ord = T_ord
        self.T_parts = T_parts
        self.theta_sat = ths.theta_sat(self.dat, self.T_ord, self.T_parts)
        self.swh = self.theta_sat.swh
        self.data_len = self.theta_sat.cAb.data_len
        self.eta_sim = self.sim_eta()

    # ===============================================================================

    def phi_sat(self, k) -> np.ndarray:
        """ Calculates the phi(k) for getting the eta(k+1) """
        u1_k = self.dat.ssd['u1'][k]
        F_k = self.dat.ssd['F'][k]
        T_k = self.dat.ssd['T'][k]
        phi_k = phiT.phi_T(T_k, self.T_ord)
        return (u1_k/F_k)*phi_k
    # ============================================================

    def sim_eta(self):
        """ Simulate the eta from data """
        eta_sim = np.zeros(self.data_len)
        for k in range(self.data_len-1):
            i = self.swh.get_interval_T(self.dat.ssd['T'][k])
            eta_sim[k+1] = ((self.phi_sat(k)).T @ self.theta_sat.thetas[self.swh.part_keys[i]])[0, 0]
        eta_sim[0] = eta_sim[1]
        return eta_sim


# Testing
if __name__ == "__main__":
    import pprint as pp
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    dat = dd.decimatedTestData(0, 2)
    sim_0 = sat_sim(dat , T_ord=0, T_parts=sh.T_none)
    sim_1 = sat_sim(dat , T_ord=1, T_parts=sh.T_none)
    sim_1w = sat_sim(dat, T_ord=1, T_parts=sh.T_wide)
    sim_1n = sat_sim(dat, T_ord=1, T_parts=sh.T_narrow)
    sim_2 = sat_sim(dat , T_ord=2, T_parts=sh.T_none)
    sim_3 = sat_sim(dat, T_ord=3, T_parts=sh.T_none)

    plt.figure()
    plt.plot(dat.ssd['t'], dat.ssd['eta'], label='eta from data set')
    # plt.plot(dat.ssd['t'], sim_0.eta_sim, label='eta_saturated_0')
    # plt.plot(dat.ssd['t'], sim_1.eta_sim, label='eta_saturated_1_none')
    # plt.plot(dat.ssd['t'], sim_1w.eta_sim, label='eta_saturated_1_wide')
    # plt.plot(dat.ssd['t'], sim_1n.eta_sim, label='eta_saturated_1_narrow')
    # plt.plot(dat.ssd['t'], sim_2.eta_sim, label='eta_saturated_2')
    plt.plot(dat.ssd['t'], sim_3.eta_sim, label='eta_saturated_3')
    plt.legend()
    plt.grid()
    plt.show()

    pp.pprint(sim_3.theta_sat.thetas)
