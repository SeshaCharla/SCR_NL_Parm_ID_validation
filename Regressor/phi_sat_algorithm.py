import numpy as np
import km_data as km
from DataProcessing import decimate_data as dd
from temperature import  phiT
# ===================

class phiSatAlg():
    """ Class holding the methods and data for phi algorithm for kth time step under saturation case"""
    #==============================================================================
    def __init__(self, dec_dat: dd.decimatedTestData, T_ord: dict) -> None:
        """ Initiates the object which holds the data set """
        self.dat = dec_dat
        self.ssd = self.dat.ssd
        self.data_len = self.dat.ssd_data_len
        self.T_ord = T_ord      # T_ord Gamma
        self.Nparms = self.T_ord['Gamma'] + 1
    # =========================================

    def phi_nox(self, k: int) -> np.ndarray:
        """ phi_sat_nox(k) =  """
        # Gaurd conditions for k ====================================
        if (k < 1) or (k >= self.data_len - 1):
            raise ValueError("k out of range of causality")
        # ===========================================================
        u1_k = self.ssd['u1'][k]
        F_k = self.ssd['F'][k]
        T_k = self.ssd['T'][k]
        phi_sat_k = (u1_k/F_k) * phiT.phi_T(T_k, self.T_ord['Gamma'])
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
            p = phiSatAlg(dat[age][test], T_ord=phiT.T_ord)
            plt.figure()
            plt.plot(p.ssd['t'][start:p.data_len-1], [p.y(k) for k in range(start, p.data_len-1)], linewidth = 1, label = p.dat.name)
            phi_sat = [p.phi_nox(k) for k in range(start, p.data_len - 1)]
            plt.figure()
            for j in range(p.Nparms-1):
                plt.plot(p.ssd['t'][start:p.data_len-1],
                         [phi_sat[k][j, 0] for k in range(len(phi_sat))],
                         linewidth = 1, label = p.dat.name)


        # ==================================================================================================================================
    plt.show()