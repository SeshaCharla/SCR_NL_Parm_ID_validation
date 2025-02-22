import numpy as np
from LS import LS_code as ls
from Regressor import km_data as km
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh


class NOx_sim():
    """ Simulating the tailpipe NOx using the hybrid model """
    def __init__(self, dat: dd.decimatedTestData):
        self.ls_sol = ls.LS_parms(dat)
        self.ssd = dat.ssd
        self.data_len = len(dat.ssd['t'])
        self.x1_sim = np.array([np.nan for i in range(self.data_len)])
        self.x1_sim[0] = self.ssd['x1'][0]
        self.x1_sim[1] = self.ssd['x1'][1]
        self.run_sim()
    # ==================================================================

    def phi_nox(self, k: int):
        """ Calculates the regression vector with the past data and inputs """
        # Gaurd conditions for k ====================================
        if (k < 1) or (k >= self.data_len - 1):
            raise ValueError("k out of range of causality")
        # ===========================================================
        x1k = self.x1_sim[k]
        kmd = km.km_dat(self.ssd, k)
        phi_k = np.matrix([[kmd.Tk], [1]])
        phi_m = np.matrix([[kmd.Tm], [1]])
        phi_nox_1 = (1 - (x1k/kmd.u1m)) * kmd.u2m * phi_m
        phi_nox_2 = (1 - (x1k/kmd.u1m)) * kmd.Fm * phi_m
        phi_nox_3 = (kmd.u1m - x1k) * kmd.Fm * phi_k
        phi_nox_4 = -(kmd.u2m/kmd.Fm) * phi_m
        phi_nox_k = np.concatenate([phi_nox_1,
                                    phi_nox_2,
                                    phi_nox_3,
                                    phi_nox_4], axis=0)
        return phi_nox_k
    # =================================================================================

    def get_interval_k(self, k: int) -> int :
        """ Get the interval of the kth time step """
        kmd = km.km_dat(self.ssd, k)
        Ti = (kmd.Tk + kmd.Tm) / 2
        i = sh.get_interval_T(Ti)
        return i
    # =====================================

    def get_theta(self, k: int) -> np.ndarray:
        """ Returns the correct parameters for the hybrid system """
        i = self.get_interval_k(k)
        return self.ls_sol.thetas[self.ls_sol.keys[i]]
    # ================================================================

    def xkp1(self, k: int):
        """ Returns xkp1 given state and inputs till k """
        theta = self.get_theta(k)
        if theta is not None:
            kmd = km.km_dat(self.ssd, k)
            x1k = self.x1_sim[k]
            eta_k = kmd.u1m - x1k
            phi_nox_k = self.phi_nox(k)
            eta_correction = (kmd.u1k/kmd.Fk)*(phi_nox_k.T @ theta)[0,0]
            x_kp1 = kmd.u1k - (eta_k * (kmd.u1k/kmd.Fk) * (kmd.Fm/kmd.u1m)) + (eta_correction)
        else:
            raise(ValueError("Theta not found"))
        return x_kp1
    # ========================================================================================================

    def run_sim(self):
        """ runs the simulation """
        for k in range(1, self.data_len-1):
            self.x1_sim[k+1] = self.xkp1(k)
    # =================================================================================================================




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    from DataProcessing import unit_convs as uc
    matplotlib.use("tkAgg")
    sim = NOx_sim(dd.decimatedTestData(1, 0))
    plt.figure()
    plt.plot(sim.ssd['t'], sim.x1_sim, label="simulated", linewidth=1)
    plt.plot(sim.ssd['t'], sim.ssd['x1'], label="data", linewidth=1)
    plt.plot(sim.ssd['t'], sim.ssd['u2'], '--', label="Urea Injection", linewidth=1)
    plt.plot(sim.ssd['t'], sim.ssd['u1'], '--', label="Inlet NOx", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("x1" + uc.units['x1'] + " | u2" + uc.units['u2'] + " | u1" + uc.units['u1'])
    plt.title(sim.ls_sol.name)
    plt.legend()
    plt.grid(True)
    plt.savefig("./figs/sim"+sim.ls_sol.name+".png")
    plt.show()
