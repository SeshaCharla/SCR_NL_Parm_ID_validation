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
        self.W0 = 1e5

        # Memoization
        self.y_vec = np.array([0] * (self.data_len-2), dtype=float)
        self.phiNox_vec = np.array([np.zeros([8,1], dtype=float)] * (self.data_len-2))
        self.P_vec = np.array([np.zeros([8, 8], dtype=float)] * (self.data_len-2))
        self.Gam_vec = np.array([np.zeros([8, 8], dtype=float)] * (self.data_len-2))
        self.pr_err_vec = np.array([0] * (self.data_len-2), dtype=float)
        self.err_pst_vec = np.array([0] * (self.data_len-2), dtype=float)
        self.mil_vec = np.array([0] * (self.data_len-2), dtype=float)
        self.theta_vec = np.array([np.zeros([8, 1], dtype=float)] * (self.data_len-2))
        self.P_eigs = np.array([0] * (self.data_len-2), dtype=float)

        # Recursion
        self.P_vec[0] = 1e2 * np.eye(8)
        self.P_eigs[0] = 1e2
        for k in range(1, self.data_len-2):
            self.recursion(k)

    # =================================================================================
    def get_phiNox(self, k: int) -> None:
        """ Calculates the kth phi vector
        """
        self.phiNox_vec[k] = self.phi_alg.phi_nox(k+1)

    # ===============================================
    def get_y(self, k: int) -> None:
        """ Calculates the kth y vector """
        self.y_vec[k] = self.phi_alg.y(k+1)

    # ===================================================================================
    def recursion(self, k: int) -> None:
        """ kth recursion of the RLS
        """
        self.get_phiNox(k)
        self.get_y(k)
        self.get_Gamma(k)
        self.get_mil_mat(k)
        self.get_P(k)
        self.get_pr_err(k)
        self.update_theta(k)
        self.get_err_pst(k)

    # ===================================================================================
    def get_Gamma(self, k: int) -> None:
        """ Calculates gamma and memoizes it
        """
        gam = 1/(self.frg_fact(k)) * self.P_vec[k-1]
        self.Gam_vec[k] = gam

    # ================================================
    def frg_fact(self, k: int) -> float:
        """ Forgetting factor for the kth time step
        """
        return 1

    # ================================================
    def get_mil_mat(self, k: int) -> None:
        """ Calculates the mil_mat and memoizes it
        """
        phi = self.phiNox_vec[k]
        gam = self.Gam_vec[k]
        mat = (1/(self.W0) + (phi.T @ gam @ phi)[0,0])
        self.mil_vec[k] = 1/mat

    # ============================================
    def get_P(self, k: int) -> None:
        """ Calculates the P and memoizes it """
        gam = self.Gam_vec[k]
        phi = self.phiNox_vec[k]
        mil = self.mil_vec[k]
        P = gam - mil*((gam @ phi) @ (phi.T @ gam))
        self.P_vec[k] = P
        self.P_eigs[k] = np.max(np.linalg.eigvals(P))

    # ==============================================
    def get_pr_err(self, k: int) -> None:
        """ Calculate prior error """
        y = self.y_vec[k]
        phi = self.phiNox_vec[k]
        th = self.theta_vec[k-1]
        self.pr_err_vec[k] = y - (phi.T @ th)[0, 0]

    # ===============================================
    def get_err_pst(self, k: int) -> None:
        """ Calculate error posterior """
        y = self.y_vec[k]
        phi = self.phiNox_vec[k]
        th = self.theta_vec[k]
        self.err_pst_vec[k] = y - (phi.T @ th)[0, 0]

    # ================================================
    def update_theta(self, k: int) -> None:
        """ Updates the theta vector """
        theta_m = self.theta_vec[k-1]
        P = self.P_vec[k]
        phi = self.phiNox_vec[k]
        p_err = self.pr_err_vec[k]
        theta = theta_m + (P @ phi) * self.W0 * p_err
        self.theta_vec[k] = theta


# ======================================================================================================================

if __name__ == '__main__':
    """ Testing """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('tkAgg')
    dat = dd.decimatedTestData(0, 1)
    rls = rls_run(dat)
    print(rls.theta_vec[-1])
    plt.figure()
    plt.plot(rls.pr_err_vec)
    plt.plot(rls.err_pst_vec)
    plt.figure()
    plt.plot(rls.P_eigs)
    plt.show()

