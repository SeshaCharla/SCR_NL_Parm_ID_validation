import numpy as np
from DataProcessing import decimate_data as dd
from DataProcessing import unit_convs as uc
import switching_handler as sh

dct = dd.decimatedTestData(1, 14)
key = 'T'
sig_T = [0, 0]
sig_T2 = [0, 0]
swh = sh.switch_handle(sh.T_hl)
print(dct.name)
for T in dct.ssd[key]:
    prt_key = swh.get_interval_T(T)
    for i in range(2):
        if prt_key == swh.part_keys[i]:
            sig_T[i] += T
            sig_T2[i] += T**2
print("sig_T = {}".format(np.array(sig_T)/dct.ssd_data_len))
print("sig_T2 = {}".format(np.array(sig_T2)/dct.ssd_data_len))
