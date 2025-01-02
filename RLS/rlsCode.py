import numpy as np
from dir_imports import  *


# =========================
class rls_run():
    """ RLS algorithm with all the memoization """
    # =============================================
    def __init__(self, dat: dd.decimatedTestData):
        """ Starts the RLS algorithm """
        self.phi_alg = ph.phiAlg(dat)
        self.name = dat.name
        self.data_len = len(dat.ssd['t'])

