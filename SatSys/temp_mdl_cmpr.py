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
T_ord = phiT.T_ord.copy()

for age in range(2):
    for tst in range(3):
        dat = dat_set[age][tst]
        T_ord["Gamma"] = 0
        sim_0   = sat_eta(dat,  T_parts=sh.T_none, T_ord=T_ord)
        sim_0w  = sat_eta(dat,  T_parts=sh.T_wide, T_ord=T_ord)
        sim_0n  = sat_eta(dat,  T_parts=sh.T_narrow, T_ord=T_ord)
        sim_0hl = sat_eta(dat,  T_parts=sh.T_hl, T_ord=T_ord)
        T_ord["Gamma"] = 1
        sim_1   = sat_eta(dat,  T_parts=sh.T_none, T_ord=T_ord)
        sim_1w  = sat_eta(dat,  T_parts=sh.T_wide, T_ord=T_ord)
        sim_1n  = sat_eta(dat,  T_parts=sh.T_narrow, T_ord=T_ord)
        sim_1hl = sat_eta(dat,  T_parts=sh.T_hl , T_ord=T_ord)
        T_ord["Gamma"] = 2
        sim_2   = sat_eta(dat,  T_parts=sh.T_none, T_ord=T_ord)
        sim_2w  = sat_eta(dat,  T_parts=sh.T_wide, T_ord=T_ord)
        sim_2hl = sat_eta(dat,  T_parts=sh.T_hl, T_ord=T_ord)

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


