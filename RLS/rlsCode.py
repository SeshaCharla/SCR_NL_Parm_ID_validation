import numpy as np
from DataProcessing import decimate_data as dd
from Regressor import phi_algorithm as ph


# =========================
class rls_run():
    """ RLS algorithm with all the memoization """
    # =============================================
    def __init__(self, dat: dd.decimatedTestData):
        """ Starts the RLS algorithm """
        self.phi_alg = ph.phiAlg(dat)
        self.name = dat.name
        self.data_len = len(dat.ssd['t'])

        # Memoization
        self.y_vec = np.array([0] * (self.data_len-2), dtype=float)
        self.phiNox_vec = np.array([np.zeros([8,1])] * (self.data_len-2), dtype=float)
        self.P_vec = np.array([np.zeros([8, 8])] * (self.data_len-2), dtype=float)
        self.Gam_vec = np.array([np.zeros([8, 8])] * (self.data_len-2), dtype=float)
        self.pr_err_vec = np.array([0] * (self.data_len-2), dtype=float)
        self.err_pst_vec = np.array([0] * (self.data_len-2), dtype=float)

    # ===================================================================================
    def Gamma(self, k):
        # Calculates gamma and memoizes it
        gam = 1/(self.frg_fact(k)) * self.P_vec[k-1]
        self.Gam_vec[k] = gam
        return gam

    # ================================================
    def frg_fact(self, k):
        # Forgetting factor for the kth time step
        return 0.99

    # ================================================




