import numpy as np
from DataProcessing import decimate_data as dd
import phi_algorithm as phi_alg
from DataProcessing.decimate_data import decimatedTestData
from HybridModel import switching_handler as sh


class PhiYmats():
    """ Container for regression matrices Y and Phi for each of the hybrid states
        This class holds  a list of Phi_NOx and Y_NOx matrices for given data
    """
    def __init__(self, dec_dat: decimatedTestData) -> None:
        self.T_parts = sh.T_parts
        self.phiAlg = phi_alg.phiAlg(dec_dat)
        self.Nparms = 8
        self.T = self.phiAlg.dat.ssd['T']
        self.data_len = self.phiAlg.data_len
        self.intervals = sh.intervals
        self.Nparts = sh.Nparts       # = 7
        self.part_keys = [str(np.array(self.intervals[i])*10 + 200) for i in range(self.Nparts)]
        self.mat_row_len = self.get_mat_row_len()
        self.Phi_NOx_mats = self.get_Phi_NOx_mats()
        self.Y_NOx_mats = self.get_Y_NOx_mats()
        self.ranks, self.PE = self.check_PE()
    # ==================================================================================================================

    def get_interval_k(self, k):
        """ Get the interval of the kth time step """
        Ti = (self.T[k] + self.T[k-1])/2
        i = sh.get_interval_T(Ti)
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
    # ==================================================================================================================

    def check_PE(self, print_stuff = False):
        """ Checks PE conditions for each of the partitions"""
        ranks = [(np.linalg.matrix_rank(Phi)>=self.Nparms) if Phi is not None else None for Phi in self.Phi_NOx_mats.values()]
        PE = [(np.min(np.linalg.eigvals(Phi.T @ Phi)) >=0) if Phi is not None else None for Phi in self.Phi_NOx_mats.values()]
        # Showing rank condition and PE results
        for i in range(self.Nparts):
            if ranks[i] is not None:
                if not ranks[i]:
                    print(self.phiAlg.dat.name + " data length is not enough for T range: " + self.part_keys[i] + " range (rank < 8)")
            if PE[i] is not None:
                if not PE[i]:
                    print("The PE condition is not satisfied for the " + self.phiAlg.dat.name + " data in the T range: " + self.part_keys[i])
        if print_stuff:
            print(ranks)
            print(PE)
        return ranks, PE
    # ==================================================================================================================

# Testing
if __name__ == "__main__":
    import pprint
    p = PhiYmats(dd.decimatedTestData(1, 0))
    T = 15.5