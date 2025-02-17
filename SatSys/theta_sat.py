from scipy.optimize import linprog
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh
import phi_sat_mats as psm


class theta_sat:
    """ Class holding the saturated system parameters for the temperature ranges """

    def __init__(self, dec_dat: dd.decimatedTestData):
        """ Loads all the data and solves the linear program in each case """
        self.dat  = dec_dat
        self.cAb = psm.cAb_mats(self.dat)
        self.thetas = self.get_thetas()
    # =========================================================================

    def get_thetas(self):
        """ Calculates the solution for each hybrid model case """
        thetas = dict()
        for key in sh.part_keys:
            if self.cAb.c_vecs[key] is not None:
                c = self.cAb.c_vecs[key]
                A = self.cAb.A_mats[key]
                b = self.cAb.b_vecs[key]
                sol = linprog(c, -A, -b, bounds=(None, None))
                # print(sol)
                thetas[key] = sol.x
            else:
                thetas[key] = None
        return thetas



# Testing
if __name__ == '__main__':
    import pprint as pp
    dat = dd.decimatedTestData(1, 0)
    thetas = theta_sat(dat)
    pp.pprint(thetas.thetas)