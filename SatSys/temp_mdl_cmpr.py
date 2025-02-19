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

for age in range(2):
    for tst in range(3):
        dat = dat_set[age][tst]
        sim_0 = sat_eta(dat, T_ord=0, T_parts=sh.T_none)
        sim_0w = sat_eta(dat, T_ord=0, T_parts=sh.T_wide)
        sim_0n = sat_eta(dat, T_ord=0, T_parts=sh.T_narrow)
        sim_0hl = sat_eta(dat, T_ord=0, T_parts=sh.T_hl)
        sim_1 = sat_eta(dat, T_ord=1, T_parts=sh.T_none)
        sim_1w = sat_eta(dat, T_ord=1, T_parts=sh.T_wide)
        sim_1n = sat_eta(dat, T_ord=1, T_parts=sh.T_narrow)
        sim_1hl = sat_eta(dat, T_ord=1, T_parts=sh.T_hl)
        sim_2 = sat_eta(dat, T_ord=2, T_parts=sh.T_none)
        sim_2w = sat_eta(dat, T_ord=2, T_parts=sh.T_wide)
        sim_2hl = sat_eta(dat, T_ord=2, T_parts=sh.T_hl)

        plt.figure()
        plt.plot(dat.ssd['t'], dat.ssd['eta'], label=r'$\eta$')
        # plt.plot(dat.ssd['t'], sim_0.eta_sim, label='eta_saturated_0_none')
        # plt.plot(dat.ssd['t'], sim_0hl.eta_sim, label='eta_saturated_0_hl')
        # plt.plot(dat.ssd['t'], sim_0w.eta_sim, label='eta_saturated_0_wide')
        # plt.plot(dat.ssd['t'], sim_0n.eta_sim, label='eta_saturated_0_narrow')
        # plt.plot(dat.ssd['t'], sim_1.eta_sim, label='eta_saturated_1_none')
        # plt.plot(dat.ssd['t'], sim_1w.eta_sim, label='eta_saturated_1_wide')
        # plt.plot(dat.ssd['t'], sim_1n.eta_sim, label='eta_saturated_1_narrow')
        # plt.plot(dat.ssd['t'], sim_2.eta_sim, label='eta_saturated_2_none')
        # plt.plot(dat.ssd['t'], sim_2w.eta_sim, label='eta_saturated_2_wide')
        plt.plot(dat.ssd['t'], sim_2hl.eta_sim, label=r'$\eta_{saturated}$')
        # plt.plot(dat.ssd['t'], dat.ssd['F'], '--', label='F')

        plt.legend()
        plt.grid()
        plt.title(dat.name)
        plt.xlabel('t [s]')
        plt.ylabel(r'$\eta$' + uc.units['eta'])
plt.show()


