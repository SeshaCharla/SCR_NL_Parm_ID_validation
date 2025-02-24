import numpy as np
from DataProcessing import decimate_data as dd
from Regressor import phi_mats as ph
from temperature import phiT
from scipy.optimize import lsq_linear
import pprint as pp
from HybridModel import switching_handler as sh

class LS_parms():
    """ Least-squares solutions for test parameter estimation """
    # ==============================================================
    def __init__(self, dat: dd.decimatedTestData, T_parts: list, T_ords:dict) -> None:
        """ Initiates the solver """
        self.dat = dat
        self.T_ords = T_ords
        self.T_parts = T_parts
        self.regr = ph.PhiYmats(self.dat, T_parts=self.T_parts, T_ords=self.T_ords)
        self.thetas = self.solve_LS()

    # =======================================================

    def solve_LS(self) -> dict:
        """ Solves the hybrid least squares problem """
        thetas = dict()
        for key_T in self.regr.swh.part_keys:
            thetas[key_T] = dict()
            for key_sat in self.regr.st_keys:
                phi = self.regr.Phi_NOx_mats[key_T][key_sat]
                y = self.regr.Y_NOx_mats[key_T][key_sat]
                if  phi is not None:
                    sol = lsq_linear(phi, y.flatten())
                    thetas[key_T][key_sat] = sol.x
                else:
                    thetas[key_T][key_sat] = None
        return thetas

    # ==================================================================
    def __repr__(self) -> str:
        """ For printing the result """
        s1 = self.dat.name
        s2 = repr(self.thetas)
        return '\n' + s1 + ': \n' + s2 + '\n'

# ======================================================================================================================
if __name__ == '__main__':
    """ Testing """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('tkAgg')

    dats = dd.load_decimated_test_data_set()
    ls_sol = [[LS_parms(dats[age][tst], T_parts=sh.T_hl, T_ords=phiT.T_ord) for tst in range(3)] for age in range(2)]
    print(ls_sol)

    # theta_names = [r'$\theta_{ads}$', r'$\theta_{od}$', r'$\theta_{scr}$', r'$\theta_{ads/scr}$']
    # sc_markers = ['s', 'x']
    # for i in range(4):
    #     plt.figure()
    #     for age in range(2):
    #         for tst in range(3):
    #             for key in ls_sol[age][tst].keys:
    #                 if ls_sol[age][tst].thetas[key] is not None:
    #                     plt.scatter(ls_sol[age][tst].thetas[key][2*i][0,0], ls_sol[age][tst].thetas[key][2*i + 1][0,0], label=ls_sol[age][tst].name, marker=sc_markers[age])
    #     plt.title(theta_names[i])
    #     plt.xlabel('m')
    #     plt.ylabel('c')
    #     plt.grid(True)
    #     plt.legend()
    #
    # plt.show()
