import numpy as np
from scipy.optimize import lsq_linear
from DataProcessing import decimate_data as dd
from Regressor import phi_algorithm as ph

class LS_solve():
    """ Least-squares solver class for test parameter estimation """
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
        Phi = np.zeros([data_len-2, 12], dtype=float)
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
    theta_names = [r'$\theta_{ads}$', r'$\theta_{od}$', r'$\theta_{scr}$', r'$\theta_{ads/scr}$']
    for i in range(4):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for age in range(2):
            for tst in range(3):
                print(ls_sol[age][tst].theta)
                ax.scatter(ls_sol[age][tst].theta[3*i], ls_sol[age][tst].theta[3*i+1], ls_sol[age][tst].theta[3*i+2],
                           label=ls_sol[age][tst].name)
                ax.set_title(theta_names[i])
                ax.grid(True)
                ax.legend()

    plt.show()

