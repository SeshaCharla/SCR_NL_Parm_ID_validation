import numpy as np
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh
from sat_sim import sat_eta
import pprint as pp
import matplotlib.pyplot as plt
import matplotlib

from temperature import phiT

matplotlib.use('TkAgg')
from DataProcessing import unit_convs as uc


dat_set = dd.load_decimated_test_data_set()

for tst in range(3):
    plt.figure()
    for age in range(2):
        dat = dat_set[age][tst]
        sim = sat_eta(dat, T_parts=sh.T_hl, T_ord=phiT.T_ord)
        plt.plot(dat.ssd['t'], sim.str_frac, label=dat.name)

    plt.legend()
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel('Storage Fraction')

for tst in range(3):
    plt.figure()
    for age in range(2):
        dat = dat_set[age][tst]
        sim = sat_eta(dat, T_parts=sh.T_hl, T_ord=phiT.T_ord)
        plt.plot(dat.ssd['t'], [ (dat.ssd['F'][k]/dat.ssd['u1'][k])*sim.eta_sim[k] for k in range(len(dat.ssd['t']))], label=dat.name)
        plt.plot(dat.ssd['t'], dat.ssd['T'], label="T_"+dat.name)

    plt.legend()
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel('Max. NOx Reduction')
plt.show()

