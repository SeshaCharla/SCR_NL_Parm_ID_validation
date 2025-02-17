import numpy as np
from DataProcessing import decimate_data as dd
from Regressor import phi_mats as ph
from scipy.optimize import lsq_linear

class LS_parms():
    """ Least-squares solutions for test parameter estimation """
    # ==============================================================
    def __init__(self, dat: dd.decimatedTestData) -> None:
        """ Initiates the solver """
        self.phm = ph.PhiYmats(dat)
        self.Wy = (1/self.phm.phiAlg.wy) * self.phm.phiAlg.W
        self.keys = self.phm.part_keys
        self.intervals = self.phm.intervals
        self.name = self.phm.phiAlg.dat.name
        self.Nparts = self.phm.Nparts
        self.Nparms = self.phm.Nparms
        self.ub = np.Inf*np.ones(self.Nparms)
        self.lb = -np.Inf*np.ones(self.Nparms)
        for i in range(self.Nparms):
            if i%2 == 0:
                self.lb[i] = 0
        self.bounds = [self.lb, self.ub]
        self.thetas = self.get_theta_dict()

    # =======================================================

    def get_theta_dict(self) -> dict[str, np.ndarray]:
        """ Returns the least-squares solution as a dictionary"""
        thetas = dict()
        for i in range(self.Nparts):
            if self.phm.Y_NOx_mats[self.keys[i]] is None:
                thetas[self.keys[i]] = None
            else:
                sol = lsq_linear(self.phm.Phi_NOx_mats[self.keys[i]],
                                 (np.array(self.phm.Y_NOx_mats[self.keys[i]])).flatten(),
                                 self.bounds)
                theta_scaled = np.matrix(sol.x)
                thetas[self.keys[i]] = self.Wy @ theta_scaled.T
        return thetas



# ======================================================================================================================
if __name__ == '__main__':
    """ Testing """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('tkAgg')

    dats = dd.load_decimated_test_data_set()
    ls_sol = [[LS_parms(dats[age][tst]) for tst in range(3)] for age in range(2)]

    theta_names = [r'$\theta_{ads}$', r'$\theta_{od}$', r'$\theta_{scr}$', r'$\theta_{ads/scr}$']
    sc_markers = ['s', 'x']
    for i in range(4):
        plt.figure()
        for age in range(2):
            for tst in range(3):
                for key in ls_sol[age][tst].keys:
                    if ls_sol[age][tst].thetas[key] is not None:
                        plt.scatter(ls_sol[age][tst].thetas[key][2*i][0,0], ls_sol[age][tst].thetas[key][2*i + 1][0,0], label=ls_sol[age][tst].name, marker=sc_markers[age])
        plt.title(theta_names[i])
        plt.xlabel('m')
        plt.ylabel('c')
        plt.grid(True)
        plt.legend()

    plt.show()
