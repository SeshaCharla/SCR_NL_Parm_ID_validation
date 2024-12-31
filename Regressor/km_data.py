import numpy as np
import math


class km_dat():
    """ Class holding the current and previous data set"""
    # ====================================================
    def __init__(self, ssd, k, check_integrity = False):
        if k < 1:
            raise ValueError("k must be >= 1")
        # ssd at time-step k
        self.x1k = ssd['x1'][k]
        self.x2k = ssd['x2'][k]
        self.u1k = ssd['u1'][k]
        self.u2k = ssd['u2'][k]
        self.Tk  = ssd['T'][k]
        self.Fk  = ssd['F'][k]
        self.etak = ssd['eta'][k]
        # ssd at time-step m = k-1
        self.x1m = ssd['x1'][k-1]
        self.x2m = ssd['x2'][k-1]
        self.u1m = ssd['u1'][k-1]
        self.u2m = ssd['u2'][k-1]
        self.Tm = ssd['T'][k-1]
        self.Fm = ssd['F'][k-1]
        self.etam = ssd['eta'][k-1]

        # Checking the integrity of eta
        if check_integrity:
            print('eta[k] = {} \n x1[k] = {} \n u1[m] = {}'.format(self.etak, self.x1k, self.u1m))
            print("diff = {}".format(self.etak - (self.u1m - self.x1k)))
            if not self.check_eta_integrity():
                raise ValueError("Eta integrity check failed!")

    # =====================================================
    def check_eta_integrity(self):
        """ checks if eta has the appropriate definition eta[k] = u1[m] - x1[k]"""
        return math.isclose(np.round(self.etak, 4), np.round((self.u1m - self.x1k), 4), rel_tol= 1e-5)

    # =============================================================================

# Testing
# ========================
if __name__ == '__main__':
    from dec_dat import *
    dat = dd.decimatedTestData(0, 0)
    km_data = km_dat(dat.ssd, 5, check_integrity = True)
    print(km_data)




