import numpy as np
from LS import LS_code as ls
from Regressor import km_data as km
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh


class NOx_sim():
    """ Simulating the tailpipe NOx using the hybrid model """
    def __init__(self, dat: dd.decimatedTestData, T_parts: list, T_ords: dict):
        self.dat = dat
        self.dat_len = len(self.dat.ssd['t'])
        self.T_ords = T_ords
        self.T_ord_k = self.T_ords[0]
        self.T_ord_kGamma = self.T_ords[1]
        self.T_parts = T_parts
        self.parms = ls.LS_parms(self.dat, T_parts = self.T_parts, T_ords = self.T_ords)

    def f_gama(self, k):
        """ Calculates f_Gamma(k) """
        pass

    def f_sigma(self, k):
        """ Calculates f_sigma(k) """
        pass

    # ==================================================================================================================

if __name__ == '__main__':
    pass
#     import matplotlib.pyplot as plt
#     import matplotlib
#     from DataProcessing import unit_convs as uc
#     matplotlib.use("tkAgg")
#     sim = NOx_sim(dd.decimatedTestData(1, 0s

    def run_sim(self):
        pass

    # ==================================================================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    from DataProcessing import unit_convs as uc
    matplotlib.use("tkAgg")
    sim = NOx_sim(dd.decimatedTestData(1, 0), T_ords=(1, 2), T_parts=sh.T_hl)

#     plt.figure()
#     plt.plot(sim.ssd['t'], sim.x1_sim, label="simulated", linewidth=1)
#     plt.plot(sim.ssd['t'], sim.ssd['x1'], label="data", linewidth=1)
#     plt.plot(sim.ssd['t'], sim.ssd['u2'], '--', label="Urea Injection", linewidth=1)
#     plt.plot(sim.ssd['t'], sim.ssd['u1'], '--', label="Inlet NOx", linewidth=1)
#     plt.xlabel("Time [s]")
#     plt.ylabel("x1" + uc.units['x1'] + " | u2" + uc.units['u2'] + " | u1" + uc.units['u1'])
#     plt.title(sim.ls_sol.name)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("./figs/sim"+sim.ls_sol.name+".png")
#     plt.show()
