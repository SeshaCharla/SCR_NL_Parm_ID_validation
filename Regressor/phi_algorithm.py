import numpy as np
from dec_dat import *
import km_data as km
# ===================

class phiAlg():
    """ Class holding the methods and data for phi algorithm for kth time step """
    #==============================================================================
    def __init__(self, dec_dat: dd.decimatedTestData):
        """ Initiates the object which holds the data set """
        self.dat = dec_dat
        self.ssd = self.dat.ssd

    # ========================================================
    def get_km_dat(self, k) -> km.km_dat:
        """ Extract the current and previous data for the time step and store it in cp_dat object """
        return km.km_dat(self.ssd, k)

    # ===============================================================================================
    def phi(self, k):
        """ phi(k) = [T[k], 1]^T """
        Tk = self.ssd["T"][k]
        phi_k = np.matrix([[Tk], [1]])
        return phi_k
