import numpy as np
import phi_algorithm as ph
from DataProcessing import decimate_data as dd
from pprint import pprint


dat = dd.load_decimated_test_data_set()
phi_dats = [[ph.phiAlg(dat[age][test]) for test in range(3)] for age in range(2)]
for age in range(2):
    for test in range(3):
        print(phi_dats[age][test].dat.name)
        pprint(np.sort((phi_dats[age][test]).check_PE()))

