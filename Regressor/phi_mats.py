import numpy as np
from DataProcessing import decimate_data as dd
import phi_algorithm as phi_alg
import phi_sat_algorithm as phi_sat_alg
from DataProcessing.decimate_data import decimatedTestData
from HybridModel import switching_handler as sh
from SatSys import sat_sim


class PhiYmats():
    """ Container for regression matrices Y and Phi for each of the hybrid states
        This class holds  a list of Phi_NOx and Y_NOx matrices for given data
    """
    def __init__(self, dec_dat: decimatedTestData) -> None:
        self.dat = dec_dat
        self.T_parts = sh.T_hl
        self.swh = sh.switch_handle(self.T_parts)
        self.eta_sat = (sat_sim.sat_eta(self.dat, T_ord=2, T_parts=self.T_parts)).eta_sim
        # =================================================================================
        self.T = self.dat.ssd['T']
        self.data_len = len(self.T)
        # ==========================
        self.st_keys = ['Sat', 'uSat']
        self.alg = dict()
        self.alg['uSat'] = phi_alg.phiAlg(self.dat)
        self.alg['Sat'] = phi_sat_alg.phiSatAlg(self.dat)
        self.mat_row_len = self.get_mat_row_len()
        self.Phi_NOx_mats = self.get_Phi_NOx_mats()
        # self.Y_NOx_mats = self.get_Y_NOx_mats()
        # self.ranks, self.PE = self.check_PE()
    # ==================================================================================================================

    def check_saturation(self, k:int) -> str:
        """ True if the kth eta is from saturation or not """
        diff = np.abs(self.eta_sat[k] - self.dat.ssd['eta'][k])
        if (diff <= 2):
            return 'Sat'
        else:
            return 'uSat'
    #=============================================================

    def get_interval_k(self, k):
        """ Get the interval of the kth time step """
        Ti = (self.T[k] + self.T[k-1])/2
        i = self.swh.get_interval_T(Ti)
        return i
    # ==================================================================================================================

    def get_mat_row_len(self) -> dict[str, dict[str, int]]:
        """ The row length of each of the regression matrices of the switched system """
        mat_sizes = dict()
        for key_T in self.swh.part_keys:
            mat_sizes[key_T] = dict()
            for key_sat in self.st_keys:
                mat_sizes[key_T][key_sat] = 0
        for k in range(1, self.data_len-1):
            key_T = self.swh.part_keys[self.get_interval_k(k)]
            key_sat = self.check_saturation(k)
            mat_sizes[key_T][key_sat] += 1
        return mat_sizes
    # ==================================================================================================================

    def get_Phi_NOx_mats(self) -> dict[str, dict[str, np.ndarray]]:
        """ Returns a dictionary of Phi_NOx matrices for each of the partitions and cases """
        # Creating the nested dictionary with zero matrices ============================================
        PhiNOxMats = dict()
        for key_T in self.swh.part_keys:
            PhiNOxMats[key_T] = dict()
            for key_sat in self.st_keys:
                if self.mat_row_len[key_T][key_sat] != 0:
                    PhiNOxMats[key_T][key_sat] = np.zeros([self.mat_row_len[key_T][key_sat], self.alg[key_sat].Nparms])
                else:
                    PhiNOxMats[key_T][key_sat] = None
        # ==========================================================================================================
        rc = dict()     # row_count dictionary
        for key_T in self.swh.part_keys:
            rc[key_T] = dict()
            for key_sat in self.st_keys:
                rc[key_T][key_sat] = 0
        # ======================================
        for k in range(1, self.data_len-1):
            key_sat = self.check_saturation(k)
            key_T = self.swh.part_keys[self.get_interval_k(k)]
            phi = self.alg[key_sat].phi_nox(k)
            PhiNOxMats[key_T][key_sat][rc[key_T][key_sat], :] = phi.flatten()
            rc[key_T][key_sat] += 1
        print(PhiNOxMats)
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
