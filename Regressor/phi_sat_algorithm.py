import numpy as np
import km_data as km
from DataProcessing import decimate_data as dd
from temperature import  phiT
# ===================

class phiSatAlg():
    """ Class holding the methods and data for phi algorithm for kth time step under saturation case"""
    #==============================================================================
    def __init__(self, dec_dat: dd.decimatedTestData, T_ord: int) -> None:
        """ Initiates the object which holds the data set """
        self.dat = dec_dat
        self.ssd = self.dat.ssd
        self.data_len = len(self.ssd['t'])
        self.T_ord = T_ord
    # =========================================

    def phi_sat_nox(self, k: int) -> np.ndarray:
        """ phi_sat_nox(k) =  """
        # Gaurd conditions for k ====================================
        if (k < 1) or (k >= self.data_len - 1):
            raise ValueError("k out of range of causality")
        # ===========================================================
        u1_k = self.ssd['u1'][k]
        F_k = self.ssd['F'][k]
        T_k = self.ssd['T'][k]
        phi_sat_k = (u1_k/F_k) * phiT.phi_T(T_k, self.T_ord)
        return phi_sat_k

    # ======================================================================

    def y(self, k: int) -> float:
        """ y(k) = eta(k+1) """
        # Gaurd conditions for k
        if (k < 1) or (k >= self.data_len -1):
            raise ValueError("k out of range of causality")
        # ============================
        eta_kp1 = self.ssd["eta"][k+1]
        return eta_kp1

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
            p = phiSatAlg(dat[age][test])
            plt.figure(10*test)
            plt.plot(p.ssd['t'][start:-1], [p.y(i) for i in range(start, len(p.ssd['t']) - 1)], linewidth = 1, label = p.dat.name)
            phi_nox_mats = np.concatenate([((p.phi_nox(k)).reshape([1, 8])) for k in range(start, len(p.ssd['t'])-1)], axis = 0)
            for i in range(8):
                plt.figure(10*test + 2 + i)
                plt.plot(p.ssd['t'][start:-1], phi_nox_mats[:, i], linewidth = 1, label = p.dat.name)
        # ==================================================================================================================================
        plt.figure(10*test)
        plt.xlabel('Time')
        plt.ylabel(r'$y_{NO_x}(k)$')
        plt.legend()
        plt.title('y in {}'.format(tsts[test]))
        plt.grid(True)
        plt.savefig("figs/y_{}".format(tsts[test]), dpi=fig_dpi)
        # ===================================================================
        phi_names = ['_ads_T', '_ads', '_od_T', '_od', '_scr_T', '_scr', 'scr/ads_T', 'scr/ads']
        for i in range(8):
            plt.figure(10*test + 2 + i)
            plt.xlabel('Time')
            plt.ylabel(r'$\phi_{NO_x}$'+'{}'.format(phi_names[i]))
            plt.legend()
            plt.title(r'$\phi_{NO_x}$'+'[:, {}] in {}'.format(i, tsts[test]))
            plt.grid(True)
            plt.savefig("figs/phi_{}_{}".format(i, tsts[test]), dpi=fig_dpi)

    plt.close('all')