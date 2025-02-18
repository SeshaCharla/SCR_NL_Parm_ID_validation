import numpy as np
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh
from temperature import phiT


class cAb_mats():
    """
        Container for the cost and constraint matrices for the linear program under various temperature bounds
    """
    def __init__(self, dec_dat: dd.decimatedTestData, T_ord:int = 1, T_parts:list = sh.T_narrow) -> None:
        self.dat = dec_dat
        self.T_ord = T_ord
        self.swh = sh.switch_handle(T_parts)
        self.Nparms = self.T_ord + 1
        self.T = self.dat.ssd['T']
        self.data_len = len(self.T)
        self.row_len = self.get_row_len()
        self.b_vecs = self.get_b()
        self.A_mats = self.get_A()
        self.c_vecs = self.get_c()
    # ==================================================================================================================
    def get_interval_k(self, k):
        """ Get the interval of the kth time step """
        i = self.swh.get_interval_T(self.T[k])
        return i
    # ==================================================================================================================

    def get_row_len(self) -> np.ndarray:
        """ The row length of each of the regression matrices of the switched system """
        mat_sizes = np.zeros(self.swh.Nparts, dtype=int)
        for k in range(1, self.data_len-1):
            i = self.get_interval_k(k)
            mat_sizes[i] += 1
        return mat_sizes
    # ==================================================================================================================

    def get_b(self) -> dict[str, np.ndarray]:
        """ Returns a dictionary of b vectors for each of the partitions """
        # Creating the dictionary with zero matrices ============================================
        b_vecs = dict()
        for i in range(self.swh.Nparts):
            if self.row_len[i] == 0:
                b_vecs[self.swh.part_keys[i]] = None
            else:
                b_vecs[self.swh.part_keys[i]] = np.zeros(self.row_len[i])
        # ========================================================================================
        irc = np.zeros(self.swh.Nparts, dtype=int)     # interval row counter
        for k in range(1, self.data_len-1):
            i = self.get_interval_k(k)
            b_vecs[self.swh.part_keys[i]][irc[i]] = self.dat.ssd['eta'][k+1]
            irc[i] += 1
        return b_vecs
    # ==================================================================================================================

    def get_A(self) -> dict[str, np.ndarray]:
        """ Returns a dictionary of A matrices for each of the partitions """
        # Creating a dictionary with zero matrices
        A_mats = dict()
        for i in range(self.swh.Nparts):
            if self.row_len[i] == 0:
                A_mats[self.swh.part_keys[i]] = None
            else:
                A_mats[self.swh.part_keys[i]] = np.zeros([self.row_len[i], self.Nparms])
        # ==========================================================================
        irc = np.zeros(self.swh.Nparts, dtype=int)
        for k in range(1, self.data_len-1):
            i = self.get_interval_k(k)
            u1_k = self.dat.ssd['u1'][k]
            F_k = self.dat.ssd['F'][k]
            T_k = self.dat.ssd['T'][k]
            A_mats[self.swh.part_keys[i]][irc[i], :] = (u1_k/F_k) * (phiT.phi_T(T_k, self.T_ord)).flatten()
            irc[i] += 1
        return A_mats
    # ==================================================================================================================

    def get_c(self) -> dict[str, np.ndarray]:
        """ Returns the c vectors for the linear programming """
        c_vecs = dict()
        for key in self.swh.part_keys:
            if self.A_mats[key] is not None:
                c_vecs[key] = np.sum(self.A_mats[key], axis=0)
            else:
                c_vecs[key] = None
        return c_vecs


# Testing
if __name__ == "__main__":
    import pprint
    cAb_rmc = cAb_mats(dd.decimatedTestData(0, 2))
    pprint.pprint(cAb_rmc.A_mats)
    pprint.pprint(cAb_rmc.c_vecs)