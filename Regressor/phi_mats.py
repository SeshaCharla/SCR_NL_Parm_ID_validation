import numpy as np
from pylint.checkers.utils import is_none

from DataProcessing import decimate_data as dd
import phi_algorithm as phi_alg
from DataProcessing.decimate_data import decimatedTestData


class PhiYmats():
    """ Container for regression matrices Y and Phi for each of the hybrid states
        This class holds  a list of Phi_NOx and Y_NOx matrices for given data
    """
    def __init__(self, dec_dat: decimatedTestData) -> None:
        self.T_parts = [-17.45, (-17.45 -6.55)/2,  -6.55, (-6.55 + 2.4)/2, 2.4, (2.4 + 15.5)/2, 15.5]
        self.phiAlg = phi_alg.phiAlg(dec_dat)
        self.Nparms = 8
        self.T = self.phiAlg.dat.ssd['T']
        self.data_len = self.phiAlg.data_len
        self.intervals = [(self.T_parts[i], self.T_parts[i+1]) for i in range(len(self.T_parts)-1)]
        self.Nparts = len(self.intervals)
        self.part_keys = [str(self.intervals[i]) for i in range(self.Nparts)]
        self.mat_row_len = self.get_mat_row_len()
        self.Phi_NOx_mats = self.get_Phi_NOx_mats()
        self.Y_NOx_mats = self.get_Y_NOx_mats()
    # ==================================================================================================================

    def get_interval_T(self, T: float) -> int:
        """ The intervals are treated as half-open on the higher side i.e., [a, b)"""
        for i in range(self.Nparts):
            if self.intervals[i][0] <= T <= self.intervals[i][1]: # this returns the first interval it belongs to unless
                return i                                            # the last value
    #===================================================================================================================

    def get_interval_k(self, k):
        """ Get the interval of the kth time step """
        Ti = (self.T[k] + self.T[k-1])/2
        i = self.get_interval_T(Ti)
        return i
    # ==================================================================================================================

    def get_mat_row_len(self) -> np.ndarray:
        """ The row length of each of the regression matrices of the switched system """
        mat_sizes = np.zeros(self.Nparts, dtype=int)
        for k in range(1, self.data_len-1):
            i = self.get_interval_k(k)
            mat_sizes[i] += 1
        return mat_sizes
    # ==================================================================================================================

    def get_Phi_NOx_mats(self) -> dict[str, np.ndarray]:
        """ Returns a dictionary of Phi_NOx matrices for each of the partitions """
        # Creating the dictionary with zero matrices ============================================
        PhiNOxMats = dict()
        for i in range(self.Nparts):
            if self.mat_row_len[i] == 0:
                PhiNOxMats[self.part_keys[i]] = None
            else:
                PhiNOxMats[self.part_keys[i]] = np.matrix(np.zeros([self.mat_row_len[i], self.Nparms]))
        # ========================================================================================
        irc = np.zeros(self.Nparts, dtype=int)     # interval row counter
        for k in range(1, self.data_len-1):
            phi = self.phiAlg.phi_nox(k)
            i = self.get_interval_k(k)
            PhiNOxMats[self.part_keys[i]][irc[i], :] = phi.flatten()
            irc[i] += 1
        return PhiNOxMats
    # ==================================================================================================================

    def get_Y_NOx_mats(self) -> dict[str, np.ndarray]:
        """ Returns a dictionary of Y_NOx matrices for each of the partitions """
        # Creating the dictionary with zero matrices ============================================
        Y_NOx_mats = dict()
        for i in range(self.Nparts):
            if self.mat_row_len[i] == 0:
                Y_NOx_mats[self.part_keys[i]] = None
            else:
                Y_NOx_mats[self.part_keys[i]] = np.matrix(np.zeros([self.mat_row_len[i], 1]))
        # =============================================================================================
        irc = np.zeros(self.Nparts, dtype=int)  # interval row counter
        for k in range(1, self.data_len - 1):
            y = self.phiAlg.y(k)
            i = self.get_interval_k(k)
            Y_NOx_mats[self.part_keys[i]][irc[i], :] = y
            irc[i] += 1
        return Y_NOx_mats


# Testing
if __name__ == "__main__":
    p = PhiYmats(dd.decimatedTestData(0, 2))
    T = 15.5
    print(p.Y_NOx_mats)