import numpy as np
from dec_dat import *
import km_data as km
# ===================

class phiAlg():
    """ Class holding the methods and data for phi algorithm for kth time step """
    #==============================================================================
    def __init__(self, dec_dat: dd.decimatedTestData) -> None:
        """ Initiates the object which holds the data set """
        self.dat = dec_dat
        self.ssd = self.dat.ssd
        self.data_len = len(self.ssd['t'])

    # ========================================================
    def get_km_dat(self, k: int) -> km.km_dat:
        """ Extract the current and previous data for the time step and store it in cp_dat object """
        return km.km_dat(self.ssd, k)

    # ===============================================================================================
    def phi(self, k: int) -> np.ndarray[2, 1]:
        """ phi(k) = [T[k], 1]^T """
        T_k = self.ssd["T"][k]
        phi_k = np.matrix([[T_k], [1]])
        return phi_k

    # =================================
    def phi_tau(self, k: int) -> np.ndarray[2, 1]:
        """ phi_tau(k) = (1/F(k)) phi(k)"""
        phi_k = self.phi(k)
        F_k = self.ssd["F"][k]
        phi_tau_k = (1/F_k) * phi_k
        return phi_tau_k

    # =======================================
    def phi_ur(self, k: int) -> np.ndarray[2, 1]:
        """ phi_ur(k) = u_2(k) * phi_tau(k) """
        phi_tau_k = self.phi_tau(k)
        u2_k = self.ssd['u2'][k]
        phi_ur_k = u2_k * phi_tau_k
        return phi_ur_k

    # ===========================================
    def phi_1(self, k: int) -> np.ndarray[2, 1]:
        """ phi_1(k) = u1(k) * phi_tau(k) """
        phi_tau_k = self.phi_tau(k)
        u1_k = self.ssd['u1'][k]
        phi_1_k = u1_k * phi_tau_k
        return phi_1_k

    # =========================================
    def f_phi1(self, k: int) -> float:
        """ f_phi1 = (uk/um) * (Fm/Fk) * (Tk*Tm +1)/(Tm^2 + 1) """
        ssd_km = self.get_km_dat(k)
        u1_frac = ssd_km.u1k/ssd_km.u1m
        F_frac = ssd_km.Fm/ssd_km.Fk
        T_frac = ((ssd_km.Tk * ssd_km.Tm) + 1) / ((ssd_km.Tm**2) + 1)
        f_phi1_k = u1_frac * F_frac * T_frac
        return f_phi1_k

    # ==============================================================
    def phi_f1(self, k: int) -> np.ndarray[6, 1]:
        """ phi_f1(k) = eta(k) * f_phi1(k) * [phi_ur(m); phi(m); u1(m)*phi(m)] """
        m = k-1
        if m < 0:
            raise ValueError("No causally preceding data")
        # =========================================
        eta_k = self.ssd["eta"][k]
        f_phi1_k = self.f_phi1(k)
        phi_ur_m = self.phi_ur(m)
        phi_m = self.phi(m)
        u1_m = self.ssd['u1'][m]
        K = eta_k * f_phi1_k
        M = np.concatenate((phi_ur_m, phi_m, u1_m*phi_m), axis=0)
        return K*M

    # ===============================================================================
    def phi_gamma1(self, k: int) -> np.ndarray[2, 1]:
        """ phi_gamma1(k) = (u1k/Fk)*(u2m/Fm)*phi(k) """
        phi_k = self.phi(k)
        km_ssd = self.get_km_dat(k)
        phi_gamma1_k = (km_ssd.u1k/km_ssd.Fk) * (km_ssd.u2m/km_ssd.Fm) * phi_k
        return phi_gamma1_k

    # ==========================================================================
    def phi_nox(self, k: int) -> np.ndarray[8, 1]:
        """ phi_nox(k) = [-phi_f1(k); phi_gamma1(k)] """
        phi_f1_k = self.phi_f1(k)
        phi_gamma1_k = self.phi_gamma1(k)
        phi_nox_k = np.concatenate((-phi_f1_k, phi_gamma1_k), axis=0)
        return phi_nox_k

    # ======================================================================
    def y(self, k: int) -> float:
        """ y(k) = eta(k+1) - eta(k)*f_phi1(k)"""
        # Gaurd conditions for k
        if (k < 1) or (k >= self.data_len -1):
            raise ValueError("k out of range of causality")
        # ============================
        eta_kp1 = self.ssd["eta"][k+1]
        eta_k = self.ssd["eta"][k]
        f_phi1_k = self.f_phi1(k)
        y_k = eta_kp1 - (eta_k * f_phi1_k)
        return y_k

    # =============================================


# ======================================================================================================================
# Testing

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('tkAgg')
    fig_dpi = 300
    tsts = ['cFTP', 'hFTP', 'RMC']

    dat = dd.load_decimated_test_data_set()
    start = 1
    for test in range(3):
        for age in range(2):
            p = phiAlg(dat[age][test])
            plt.figure(10*test)
            plt.plot(p.ssd['t'][start:-1], [p.y(i) for i in range(start, len(p.ssd['t']) - 1)], linewidth = 1, label = p.dat.name)
            plt.figure(10*test+1)
            plt.plot(p.ssd['t'][start:], [p.f_phi1(i) for i in range(start, len(p.ssd['t']))], linewidth = 1, label = p.dat.name)
            phi_nox_mats = np.concatenate([((p.phi_nox(k)).reshape([1, 8])) for k in range(start, len(p.ssd['t']))], axis = 0)
            for i in range(8):
                plt.figure(10*test + 2 + i)
                plt.plot(p.ssd['t'][start:], phi_nox_mats[:, i], linewidth = 1, label = p.dat.name)
        # ==================================================================================================================================
        plt.figure(10*test)
        plt.xlabel('Time')
        plt.ylabel(r'$y(k) = \eta(k+1) - f_{\phi1}(k) * \eta(k)$')
        plt.legend()
        plt.title('y in {}'.format(tsts[test]))
        plt.grid(True)
        plt.savefig("figs/y_{}".format(tsts[test]), dpi=fig_dpi)
        plt.close()
        # ===================================================================
        plt.figure(10*test+1)
        plt.xlabel('Time')
        plt.ylabel(r'$f_{\phi1}$')
        plt.title(r'$f_{\phi1}$ in ' + tsts[test])
        plt.legend()
        plt.grid(True)
        plt.savefig("figs/f_{}".format(tsts[test]), dpi=fig_dpi)
        plt.close()
        for i in range(8):
            plt.figure(10*test + 2 + i)
            plt.xlabel('Time')
            plt.ylabel(r'$\phi_{NO_x}$'+'[:, {}]'.format(i))
            plt.legend()
            plt.title(r'$\phi_{NO_x}$'+'[:, {}] in {}'.format(i, tsts[test]))
            plt.grid(True)
            plt.savefig("figs/phi_{}_{}".format(i, tsts[test]), dpi=fig_dpi)
            plt.close()
