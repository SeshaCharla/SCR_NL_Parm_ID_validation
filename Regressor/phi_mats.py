import numpy as np
from DataProcessing import decimate_data as dd
from Regressor import phi_algorithm as phi_alg
from Regressor import phi_sat_algorithm as phi_sat_alg
from DataProcessing.decimate_data import decimatedTestData
from HybridModel import switching_handler as sh
from SatSys import sat_sim
import  pprint as pp


class PhiYmats():
    """ Container for regression matrices Y and Phi for each of the hybrid states
        This class holds  a list of Phi_NOx and Y_NOx matrices for given data
    """
    def __init__(self, dec_dat: decimatedTestData,  T_parts: list, T_ords: tuple,) -> None:
        self.dat = dec_dat
        self.T_parts = T_parts
        self.swh = sh.switch_handle(self.T_parts)
        self.eta_sat = (sat_sim.sat_eta(self.dat, T_ord=T_ords[1], T_parts=self.T_parts)).eta_sim
        self.T_ord_k = T_ords[0]
        self.T_ord_kGamma = T_ords[1]
        # =================================================================================
        self.T = self.dat.ssd['T']
        self.data_len = len(self.T)
        # ==========================
        self.st_keys = ['Sat', 'uSat']
        self.alg = dict()
        self.alg['uSat'] = phi_alg.phiAlg(self.dat, T_ord_k=self.T_ord_k, T_ord_kGamma=self.T_ord_kGamma)
        self.alg['Sat'] = phi_sat_alg.phiSatAlg(self.dat, T_ord_kGamma=self.T_ord_kGamma)
        self.mat_row_len = self.get_mat_row_len()
        self.Phi_NOx_mats = self.get_Phi_NOx_mats()
        self.Y_NOx_mats = self.get_Y_NOx_mats()
        self.ranks, self.PE = self.check_PE(print_stuff=False)
    # ==================================================================================================================

    def check_saturation(self, k:int) -> str:
        """ True if the kth eta is from saturation or not """
        diff = np.abs(self.eta_sat[k] - self.dat.ssd['eta'][k])
        if (diff <= 2):
            return 'Sat'
        else:
            return 'uSat'
    #=============================================================

    def get_interval_k(self, k) -> str:
        """ Get the interval of the kth time step """
        Ti = (self.T[k] + self.T[k-1])/2
        key = self.swh.get_interval_T(Ti)
        return key
    # ==================================================================================================================

    def get_mat_row_len(self) -> dict[str, dict[str, int]]:
        """ The row length of each of the regression matrices of the switched system """
        mat_sizes = dict()
        for key_T in self.swh.part_keys:
            mat_sizes[key_T] = dict()
            for key_sat in self.st_keys:
                mat_sizes[key_T][key_sat] = 0
        for k in range(1, self.data_len-1):
            key_T = self.get_interval_k(k)
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
            key_T = self.get_interval_k(k)
            phi = self.alg[key_sat].phi_nox(k)
            PhiNOxMats[key_T][key_sat][rc[key_T][key_sat], :] = phi.flatten()
            rc[key_T][key_sat] += 1
        return PhiNOxMats
    # ==================================================================================================================

    def get_Y_NOx_mats(self) -> dict[str, dict[str, np.ndarray]]:
        """ Returns a dictionary of Y_NOx matrices for each of the partitions """
        # Creating the dictionary with zero matrices ============================================
        yNOxMats = dict()
        for key_T in self.swh.part_keys:
            yNOxMats[key_T] = dict()
            for key_sat in self.st_keys:
                if self.mat_row_len[key_T][key_sat] != 0:
                    yNOxMats[key_T][key_sat] = np.zeros(self.mat_row_len[key_T][key_sat])
                else:
                    yNOxMats[key_T][key_sat] = None
        # ==========================================================================================================
        rc = dict()     # row_count dictionary
        for key_T in self.swh.part_keys:
            rc[key_T] = dict()
            for key_sat in self.st_keys:
                rc[key_T][key_sat] = 0
        # ======================================
        for k in range(1, self.data_len-1):
            key_sat = self.check_saturation(k)
            key_T = self.get_interval_k(k)
            y = self.alg[key_sat].y(k)
            yNOxMats[key_T][key_sat][rc[key_T][key_sat]] = y
            rc[key_T][key_sat] += 1
        return yNOxMats
    # ==================================================================================================================

    def check_PE(self, print_stuff = False):
        """ Checks PE conditions for each of the partitions"""
        ranks = dict()
        PE = dict()
        for key_T in self.swh.part_keys:
            ranks[key_T] = dict()
            PE[key_T] = dict()
            for key_sat in self.st_keys:
                phi = self.Phi_NOx_mats[key_T][key_sat]
                if phi is not None:
                    ranks[key_T][key_sat] = np.linalg.matrix_rank(phi)
                    PE[key_T][key_sat] = np.round(np.min(np.linalg.eigvals(phi @ phi.T)),2)
                else:
                    PE[key_T][key_sat] = None
                    ranks[key_T][key_sat] = None
        # Showing rank condition and PE results
        for key_T in self.swh.part_keys:
            for key_sat in self.st_keys:
                if ranks[key_T][key_sat] is not None:
                    if ranks[key_T][key_sat] < self.alg[key_sat].Nparms:
                        print("Data length is not enough for unique paremeters for T in range "
                                + str(key_T) + " under " + str(key_sat))
                if PE[key_T][key_sat] is not None:
                    if PE[key_T][key_sat] < 0:
                        print("PE condition is not satisfied for T in range "
                                + str(key_T) + " under " + str(key_sat))
        if print_stuff:
            pp.pprint("ranks:")
            pp.pprint(ranks)
            pp.pprint("Min Eigen Value:")
            pp.pprint(PE)
        return ranks, PE
    # ==================================================================================================================

# Testing
if __name__ == "__main__":
    import pprint
    p = PhiYmats(dd.decimatedTestData(1, 1), T_parts=sh.T_hl, T_ords=(2, 2))
