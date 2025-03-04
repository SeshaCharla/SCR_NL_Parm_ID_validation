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
        self.x1_sim, self.eta_sim, self.f_sig, self.f_gam = self.run_sim()

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

    def eta_kp1(self, x1_k, k):
        """ Calculates eta(k+1) given eta(k) """
        f_sig_k = self.f_sigma(x1_k, k)
        f_gamma_k = self.f_gamma(k)
        sig_diff = np.abs(f_sig_k - self.dat.ssd['u1'][k-1])
        gam_diff = np.abs(f_gamma_k - self.dat.ssd['u1'][k-1])
        if gam_diff < sig_diff:
            f_eta = f_gamma_k
        else:
            f_eta = f_sig_k
            if f_eta < 0:
                f_eta = 0
        return f_eta

    def run_sim(self):
        """ Runs the simulation """
        x1_sim = np.zeros(self.dat_len)
        eta_sim = np.zeros(self.dat_len)
        f_sig = np.zeros(self.dat_len)
        f_gam = np.zeros(self.dat_len)
        x1_sim[0] = self.dat.ssd['x1'][0]
        eta_sim[0] = self.dat.ssd['eta'][0]
        f_sig[0] = eta_sim[0]
        f_gam[0] = eta_sim[0]
        x1_sim[1] = self.dat.ssd['x1'][1]
        eta_sim[1] = self.dat.ssd['eta'][1]
        for k in range(1, self.dat_len-1):
            eta_sim[k+1] = self.eta_kp1(x1_sim[k], k)
            x1_sim[k+1] = np.max([0,self.dat.ssd['u1'][k] - eta_sim[k+1]])
            # Bounds
            f_sig[k] = self.f_sigma(x1_sim[k], k)
            f_gam[k] = self.f_gamma(k)
        f_sig[-1] = eta_sim[-1]
        f_gam[-1] = eta_sim[-1]
        return x1_sim, eta_sim, f_sig, f_gam

    # ==================================================================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    from DataProcessing import unit_convs as uc
    from Simulation import pfit
    matplotlib.use("tkAgg")

#    plt.figure()
#    plt.plot(sim.dat.ssd['t'], sim.x1_sim, label="simulated", linewidth=1)
#    plt.plot(sim.dat.ssd['t'], sim.dat.ssd['x1'], label="data", linewidth=1)
#    # plt.plot(sim.dat.ssd['t'], sim.dat.ssd['u2'], '--', label="Urea Injection", linewidth=1)
#    # plt.plot(sim.dat.ssd['t'], sim.dat.ssd['u1'], '--', label="Inlet NOx", linewidth=1)
#    plt.xlabel("Time [s]")
#    plt.ylabel("x1" + uc.units['x1'] + " | u2" + uc.units['u2'] + " | u1" + uc.units['u1'])
#    plt.title("Tailpipe NOx" + sim.dat.name)
#    plt.legend()
#    plt.grid(True)
#    plt.savefig("./figs/sim"+sim.dat.name+".png")

    for age in range(2):
        for test in range(3):
            sim = NOx_sim(dd.decimatedTestData(age, test), T_parts=sh.T_hl, T_ords=phiT.T_ord)
            per_fit = pfit.pfit(sim.dat.ssd['eta'], sim.eta_sim)
            if test == 0 or test == 1:
                per_fit = pfit.pfit(sim.dat.ssd['eta'][400:900], sim.eta_sim[400:900])
            s = "%fit = {}".format(np.round(per_fit, 2))
            plt.figure()
            plt.plot(sim.dat.ssd['t'], sim.eta_sim, label="simulated", linewidth=1)
            plt.plot(sim.dat.ssd['t'], sim.dat.ssd['eta'], label="data", linewidth=1)
            # plt.plot(sim.dat.ssd['t'], sim.f_sig, '--', label="simulated Unsaturated", linewidth=1)
            # plt.plot(sim.dat.ssd['t'], sim.f_gam, '--', label="simulated Saturated", linewidth=1)
            plt.text(300, 30, s)
            plt.ylim([-0.1, 50])
            plt.xlabel("Time [s]")
            plt.ylabel(r'$\eta$' + uc.units['eta'])
            plt.title("NOx reduction per-sample - " + sim.dat.name)
            plt.legend()
            plt.grid(True)
            plt.savefig("./figs/eta_sim_" + sim.dat.name + ".png")
    plt.show()