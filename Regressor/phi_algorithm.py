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
            raise ValueError("k needs to be >= 1")
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
        """ y(k) = eta(k) - eta(k-1)*f_phi1(k-1)"""
        m = k-1
        if k < 2:
            raise ValueError("k needs to be >= 2")
        # =========================================
        eta_k = self.ssd["eta"][k]
        eta_m = self.ssd["eta"][m]
        f_phi1_m = self.f_phi1(m)
        y_k = eta_k - eta_m * f_phi1_m
        return y_k

    # =============================================


# ======================================================================================================================
# Testing

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('tkAgg')

    dat = dd.load_decimated_test_data_set()
    start = 2
    for test in range(3):
        plt.figure()
        for age in range(2):
            p = phiAlg(dat[age][test])
            plt.plot(p.ssd['t'][start:], [p.y(i) for i in range(start, len(p.ssd['t']))], linewidth = 1, label = p.dat.name)
        plt.xlabel('Time')
        plt.ylabel('y = eta(k) - f_phi1(m) * eta(m)')
        plt.legend()
        plt.grid(True)
    plt.show()
