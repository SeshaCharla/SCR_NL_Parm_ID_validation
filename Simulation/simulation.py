import numpy as np
from LS import LS_code as ls
from Regressor import km_data as km
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh
from temperature import phiT


class NOx_sim():
    """ Simulating the tailpipe NOx using the hybrid model """
    def __init__(self, dat: dd.decimatedTestData, T_parts: list, T_ords: dict):
        self.dat = dat
        self.dat_len = self.dat.ssd_data_len
        self.T_ords = T_ords
        self.T_parts = T_parts
        self.parms = ls.LS_parms(self.dat, T_parts = self.T_parts, T_ords = self.T_ords)
        self.thetas = self.parms.thetas
        self.swh = sh.switch_handle(self.T_parts)

    def f_sigma(self, k):
        """ Calculates f_sigma(k) """
        key_T = self.swh.get_interval_T(self.dat.ssd['T'][k])
        theta_NOx = self.thetas[key_T]['uSat']
        pass

    def f_gamma(self, k):
        """ Calculates f_Gamma(k) """
        kmd = km.km_dat(self.dat.ssd, k)
        key_T = self.swh.get_interval_T(self.dat.ssd['T'][k])
        theta_Gamma = self.thetas[key_T]['Sat']
        f_gam = (kmd.u1k/kmd.Fk) * (theta_Gamma @ phiT.phi_T(kmd.Tk, self.T_ords['Gamma']))[0, 0]
        return f_gam

    # ==================================================================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    from DataProcessing import unit_convs as uc
    matplotlib.use("tkAgg")
    sim = NOx_sim(dd.decimatedTestData(0, 2), T_parts=sh.T_hl, T_ords=phiT.T_ord)
    print(sim.f_gamma(200))

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
