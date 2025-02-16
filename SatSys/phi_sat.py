import numpy as np
from Regressor import  km_data as  km
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
        self.W = np.diag([1e-1, 1, 1e-2, 1e-1, 1e-3, 1e-2, 1, 10])
                        # 0   , 1, 2   , 3   , 4   , 5   , 6 , 7
        self.wy = 10

    # ========================================================
    def get_km_dat(self, k: int) -> km.km_dat:
        """ Extract the current and previous data for the time step and store it in cp_dat object """
        return km.km_dat(self.ssd, k)

    # =========================================
    def phi_nox(self, k: int) -> np.ndarray[8, 1]:
        """ phi_nox(k) = [-phi_eta(k); phi_Gamma(k)] """
        # Gaurd conditions for k ====================================
        if (k < 1) or (k >= self.data_len - 1):
            raise ValueError("k out of range of causality")
        # ===========================================================
        km = self.get_km_dat(k)
        x1_u1_1 = (km.x1k/km.u1m) - 1
        phi_k = np.matrix([[km.Tk], [1]])
        phi_m = np.matrix([[km.Tm], [1]])
        phi_nox_1 = x1_u1_1 * km.u2m * phi_m
        phi_nox_2 = x1_u1_1 * km.Fm * phi_m
        phi_nox_3 = -km.etak * km.Fm * phi_k
        phi_nox_4 = (km.u2m/km.Fm) * phi_k
        phi_nox_k = np.concatenate([phi_nox_1,
                                    phi_nox_2,
                                    phi_nox_3,
                                    phi_nox_4], axis=0)
        return self.W @ phi_nox_k

    # ======================================================================
    def y(self, k: int) -> float:
        """ y(k) = Fu1(k) * eta(k+1) - eta(k)*f_phi1(k)"""
        # Gaurd conditions for k
        if (k < 1) or (k >= self.data_len -1):
            raise ValueError("k out of range of causality")
        # ============================
        eta_kp1 = self.ssd["eta"][k+1]
        km = self.get_km_dat(k)
        y_k = ((km.Fk/km.u1k) * eta_kp1) - ((km.Fm/km.u1m) * km.etak)
        return self.wy * y_k

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