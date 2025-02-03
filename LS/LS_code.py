import numpy as np
from DataProcessing import decimate_data as dd
from Regressor import phi_mats as ph

class LS_parms():
    """ Least-squares solutions for test parameter estimation """
    # ==============================================================
    def __init__(self, dat: dd.decimatedTestData) -> None:
        """ Initiates the solver """
        self.dat = dat
        self.name = dat.name
        self.phi_alg = ph.phiAlg(dat)
        self.Phi = self.make_phi_mat()
        self.Y = self.make_Y_mat()
        self.sol = self.con_lsq_solve()
        self.theta = self.sol[0]

    # =====================================================
    def make_phi_mat(self) -> np.ndarray:
        """ Creat the Phi matrix """
        data_len = self.phi_alg.data_len
        Phi = np.zeros([data_len-2, 8], dtype=float)
        for k in range(data_len-2):
            Phi[k, :] = (self.phi_alg.phi_nox(k+1)).T
        return Phi

    # =====================================================
    def make_Y_mat(self) -> np.ndarray:
        """ Creat the Y matrix """
        data_len = self.phi_alg.data_len
        Y = np.zeros(data_len-2, dtype=float)
        for k in range(data_len-2):
            Y[k] = self.phi_alg.y(k+1)
        return Y

    # =======================================================
    def con_lsq_solve(self):
        """" Solve the constrained least-squares problem """
        sol = np.linalg.lstsq(self.Phi, self.Y)
        return sol

    # =======================================================

# ======================================================================================================================
if __name__ == '__main__':
    """ Testing """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('tkAgg')

    dats = dd.load_decimated_test_data_set()
    ls_sol = [[LS_solve(dats[age][tst]) for tst in range(3)] for age in range(2)]

    # Mixed data solution
    ls_mats_dg = [LS_solve(dats[0][tst]) for tst in range(3)]
    ls_mats_ag = [LS_solve(dats[1][tst]) for tst in range(3)]

    Y_dg = np.concatenate([ls_mats_dg[i].Y[600:900] for i in range(len(ls_mats_dg))])
    Phi_dg = np.concatenate([ls_mats_dg[i].Phi[600:900, :] for i in range(len(ls_mats_dg))])
    theta_dg = np.linalg.lstsq(Phi_dg, Y_dg)[0]
    print(theta_dg)

    Y_ag = np.concatenate([ls_mats_ag[i].Y[600:900] for i in range(len(ls_mats_ag))])
    Phi_ag = np.concatenate([ls_mats_ag[i].Phi[600:900, :] for i in range(len(ls_mats_ag))])
    theta_ag = np.linalg.lstsq(Phi_ag, Y_ag)[0]
    print(theta_ag)

    theta_names = [r'$\theta_{ads}$', r'$\theta_{od}$', r'$\theta_{scr}$', r'$\theta_{ads/scr}$']
    sc_markers = ['s', 'x']
    for i in range(4):
        plt.figure()
        plt.scatter(theta_dg[2 * i], theta_dg[2 * i + 1], label="degreened_mixed", marker=sc_markers[0])
        plt.scatter(theta_ag[2 * i], theta_ag[2 * i + 1], label="aged_mixed", marker=sc_markers[1])
        for age in range(2):
            for tst in range(3):
                plt.scatter(ls_sol[age][tst].theta[2*i], ls_sol[age][tst].theta[2*i+1], label=ls_sol[age][tst].name, marker=sc_markers[age])
        plt.title(theta_names[i])
        plt.xlabel('m')
        plt.ylabel('c')
        plt.grid(True)
        plt.legend()

    plt.show()
