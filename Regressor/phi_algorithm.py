import numpy as np
import km_data as km
from DataProcessing import decimate_data as dd
from temperature import  phiT
# ===================

class phiAlg():
    """ Class holding the methods and data for phi algorithm for kth time step """
    #==============================================================================
    def __init__(self, dec_dat: dd.decimatedTestData, T_ord: dict) -> None:
        """ Initiates the object which holds the data set """
        self.dat = dec_dat
        self.ssd = self.dat.ssd
        self.data_len = self.dat.ssd_data_len
        self.T_ord = T_ord
        self.Nparms = np.sum([self.T_ord[key] + 1 for key in T_ord.keys()])

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
        phi_nox_ads = x1_u1_1 * km.u2m * phiT.phi_T(km.Tm, self.T_ord['ads'])
        phi_nox_od  = x1_u1_1 * km.Fm * phiT.phi_T(km.Tm, self.T_ord['od'])
        phi_nox_scr = -km.etak * km.Fm * phiT.phi_T(km.Tk, self.T_ord['scr'])
        phi_nox_Gamma = (km.u2m/km.Fm) * phiT.phi_T(km.Tm, self.T_ord['Gamma'])
        phi_nox_k = np.concatenate([phi_nox_ads,
                                    phi_nox_od,
                                    phi_nox_scr,
                                    phi_nox_Gamma], axis=0)
        return phi_nox_k

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
        return y_k

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
            p = phiAlg(dat[age][test], T_ord=phiT.T_ord)
            plt.figure(11*test)
            plt.plot(p.ssd['t'][start:-1], [p.y(i) for i in range(start, len(p.ssd['t']) - 1)], linewidth = 1, label = p.dat.name)
            phi_nox_mats = np.concatenate([((p.phi_nox(k)).reshape([1, p.Nparms])) for k in range(start, len(p.ssd['t'])-1)], axis = 0)
            for i in range(p.Nparms):
                plt.figure(11*test + 2 + i)
                plt.plot(p.ssd['t'][start:-1], phi_nox_mats[:, i], linewidth = 1, label = p.dat.name)
        # ==================================================================================================================================
        plt.figure(11*test)
        plt.xlabel('Time')
        plt.ylabel(r'$y_{NO_x}(k)$')
        plt.legend()
        plt.title('y in {}'.format(tsts[test]))
        plt.grid(True)
        plt.savefig("figs/y_{}".format(tsts[test]), dpi=fig_dpi)
        # ===================================================================
        phi_names = ['_ads_T', '_ads',
                     '_od_T', '_od',
                     '_scr_T', '_scr',
                     'scr/ads_T^2', 'scr/ads_T', 'scr/ads']
        for i in range(p.Nparms):
            plt.figure(11*test + 2 + i)
            plt.xlabel('Time')
            plt.ylabel(r'$\phi_{NO_x}$'+'{}'.format(phi_names[i]))
            plt.legend()
            plt.title(r'$\phi_{NO_x}$'+'[:, {}] in {}'.format(i, tsts[test]))
            plt.grid(True)
            plt.savefig("figs/phi_{}_{}".format(i, tsts[test]), dpi=fig_dpi)

    plt.close('all')