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

    def f_sigma(self, x1_k, k):
        """ Calculates f_sigma(k) """
        kmd = km.km_dat(self.dat.ssd, k)
        key_T = self.swh.get_interval_T(self.dat.ssd['T'][k])
        theta_NOx = self.thetas[key_T]['uSat']
        eta_k = kmd.u1m - x1_k
        # ====================
        x1_u1_1 = (x1_k / kmd.u1m) - 1
        phi_nox_ads = x1_u1_1 * kmd.u2m * phiT.phi_T(kmd.Tm, self.T_ords['ads'])
        phi_nox_od = x1_u1_1 * kmd.Fm * phiT.phi_T(kmd.Tm, self.T_ords['od'])
        phi_nox_scr = -eta_k * kmd.Fm * phiT.phi_T(kmd.Tk, self.T_ords['scr'])
        phi_nox_Gamma = (kmd.u2m / kmd.Fm) * phiT.phi_T(kmd.Tm, self.T_ords['Gamma'])
        phi_nox_k = np.concatenate([phi_nox_ads,
                                    phi_nox_od,
                                    phi_nox_scr,
                                    phi_nox_Gamma], axis=0)
        # ====================
        scalar_term = eta_k * (kmd.u1k/kmd.Fk) * (kmd.Fm/kmd.u1m)
        matrix_product = (kmd.u1k/kmd.Fk) * (theta_NOx @ phi_nox_k)[0, 0]
        f_sig = scalar_term + matrix_product
        return f_sig

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
    k = 700
    print(sim.f_gamma(k))
    print(sim.f_sigma(sim.dat.ssd['x1'][k], k))
    print(sim.dat.ssd['u1'][k])

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
