import numpy as np
import km_data as km
from DataProcessing import decimate_data as dd
# ===================

class phiAlg():
    """ Class holding the methods and data for phi algorithm for kth time step """
    #==============================================================================
    def __init__(self, dec_dat: dd.decimatedTestData) -> None:
        """ Initiates the object which holds the data set """
        self.dat = dec_dat
        self.ssd = self.dat.ssd
        self.data_len = len(self.ssd['t'])
        self.W = np.diag([1e-3, 1e-1, 1e-3, 1e-1, 1e-5, 1e-4, 1e-1, 1e1])

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

    # =========================================
    def f_r(self, k: int) -> float:
        """ f_r = (Fm/um) * (Tk*Tm +1)/(Tm^2 + 1) """
        ssd_km = self.get_km_dat(k)
        Fu1_m = ssd_km.Fm/ssd_km.u1m
        T_frac = ((ssd_km.Tk * ssd_km.Tm) + 1) / ((ssd_km.Tm**2) + 1)
        f_phi1_k = Fu1_m * T_frac
        return f_phi1_k

    # ==============================================================
    def phi_fr(self, k: int) -> np.ndarray[6, 1]:
        """ phi_f1(k) = eta(k) * [(u2m/u1m)*phi(k); (Fm/u1m)*phi(k); (Fm) * phi(k)] """
        # =========================================
        ssd_km = self.get_km_dat(k)
        phi_k = self.phi(k)
        M1 = (ssd_km.u2m/ssd_km.u1m) * phi_k
        M2 = (ssd_km.Fm/ssd_km.u1m) * phi_k
        M3 = ssd_km.Fm*phi_k
        K = self.ssd["eta"][k]
        M = np.concatenate((M1, M2, M3), axis=0)
        return K*M

    # ===============================================================================
    def phi_Gamma_r(self, k: int) -> np.ndarray[2, 1]:
        """ phi_gamma1(k) = (u2m/Fm)*phi(k) """
        phi_k = self.phi(k)
        km_ssd = self.get_km_dat(k)
        phi_Gamma_r_k = (km_ssd.u2m/km_ssd.Fm) * phi_k
        return phi_Gamma_r_k

    # ==========================================================================
    def phi_nox(self, k: int) -> np.ndarray[8, 1]:
        """ phi_nox(k) = [-phi_f1(k); phi_gamma1(k)] """
        phi_fr_k = self.phi_fr(k)
        phi_Gamma_r_k = self.phi_Gamma_r(k)
        phi_nox_k = np.concatenate((-phi_fr_k, phi_Gamma_r_k), axis=0)
        return self.W @ phi_nox_k

    # ======================================================================
    def y(self, k: int) -> float:
        """ y(k) = Fu1(k) * eta(k+1) - eta(k)*f_phi1(k)"""
        # Gaurd conditions for k
        if (k < 1) or (k >= self.data_len -1):
            raise ValueError("k out of range of causality")
        # ============================
        eta_kp1 = self.ssd["eta"][k+1]
        eta_k = self.ssd["eta"][k]
        f_r_k = self.f_r(k)
        ssd_km = self.get_km_dat(k)
        Fu1_k = ssd_km.Fk/ssd_km.u1k
        y_k = (Fu1_k * eta_kp1) - (eta_k * f_r_k)
        return y_k

    # =============================================
    def check_PE(self):
        """ Do a PE condition check for the phi """
        sum = np.zeros([8,8])
        for k in range(1, self.data_len):
            phi_k = self.phi_nox(k)
            sum += phi_k @ phi_k.T
        return np.linalg.eigvals(sum)

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
            plt.plot(p.ssd['t'][start:], [p.f_r(i) for i in range(start, len(p.ssd['t']))], linewidth = 1, label = p.dat.name)
            phi_nox_mats = np.concatenate([((p.phi_nox(k)).reshape([1, 8])) for k in range(start, len(p.ssd['t']))], axis = 0)
            for i in range(8):
                plt.figure(10*test + 2 + i)
                plt.plot(p.ssd['t'][start:], phi_nox_mats[:, i], linewidth = 1, label = p.dat.name)
        # ==================================================================================================================================
        plt.figure(10*test)
        plt.xlabel('Time')
        plt.ylabel(r'$y(k) = F_{u_1}(k)* \eta(k+1) - f_{r}(k) * \eta(k)$')
        plt.legend()
        plt.title('y in {}'.format(tsts[test]))
        plt.grid(True)
        plt.savefig("figs/y_{}".format(tsts[test]), dpi=fig_dpi)
        plt.close()
        # ===================================================================
        plt.figure(10*test+1)
        plt.xlabel('Time')
        plt.ylabel(r'$f_{r}$')
        plt.title(r'$f_{r}$ in ' + tsts[test])
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
