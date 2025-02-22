import numpy as np
from DataProcessing import decimate_data as dd
from HybridModel import switching_handler as sh
from sat_sim import sat_eta
import pprint as pp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from DataProcessing import unit_convs as uc


dat_set = dd.load_decimated_test_data_set()

for tst in range(3):
    plt.figure()
    for age in range(2):
        dat = dat_set[age][tst]
        sim = sat_eta(dat, T_ord=2, T_parts=sh.T_hl)
        plt.plot(dat.ssd['t'], sim.str_frac, label=dat.name)

    plt.legend()
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel('Storage Fraction')

for tst in range(3):
    plt.figure()
    for age in range(2):
        dat = dat_set[age][tst]
        sim = sat_eta(dat, T_ord=2, T_parts=sh.T_hl)
        plt.plot(dat.ssd['t'], sim.eta_sim, label=dat.name)

    plt.legend()
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel('Max. NOx Reduction')
plt.show()

