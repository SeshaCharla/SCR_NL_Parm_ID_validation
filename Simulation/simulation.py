import numpy as np
from LS import LS_code as ls
from Regressor import km_data as km
from DataProcessing import decimate_data as dd


class NOx_sim():
    """ Simulating the tailpipe NOx using the hybrid model """
    def __init__(self, dat: dd.decimatedTestData):
        ls_sol = ls.LS_parms(dat)
        self.ssd = dat.ssd
        self.data_len = ls_sol.phm.data_len
        self.x1_sim = np.array([np.nan for i in range(self.data_len)])
        self.x1_sim[0] = self.ssd['x1'][0]
        print(self.x1_sim)

    def phi_nox(self, k: int):
        """ Calculates the regression vector with the past data and inputs """
        # Gaurd conditions for k ====================================
        if (k < 1) or (k >= self.data_len - 1):
            raise ValueError("k out of range of causality")
        # ===========================================================
        x1k = self.x1_sim[k]
        kmd = km.km_dat(self.ssd, k)
        phi_k = np.matrix([[kmd.Tk], [1]])
        phi_m = np.matrix([[kmd.Tm], [1]])
        phi_nox_1 = (1 - (x1k/kmd.u1m)) * kmd.u2m * phi_m
        phi_nox_2 = (1 - (x1k/kmd.u1m)) * kmd.Fm * phi_m
        phi_nox_3 = (kmd.u1m - x1k) * kmd.Fm * phi_k
        phi_nox_4 = -(kmd.u2m/kmd.Fm) * phi_m
        phi_nox_k = np.concatenate([phi_nox_1,
                                    phi_nox_2,
                                    phi_nox_3,
                                    phi_nox_4], axis=0)
        return phi_nox_k

    def get_theta(self, k: int) -> np.ndarray:
        """ Returns the correct parameters for the hybrid system """

    def xkp1(self, k: int):
        """ Returns xkp1 given state and inputs till k """
        kmd = km.km_dat(self.ssd, k)
        x1k = self.x1_sim[k]
        eta_k = kmd.u1m - x1k
        phi_nox_k = self.phi_nox(k)
        theta = self.get_theta(k)




if __name__ == '__main__':
    sim = NOx_sim(dd.decimatedTestData(0, 2))
    print(sim.phi_nox(3))