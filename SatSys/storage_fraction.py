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

# for tst in range(3):
#     plt.figure()
#     for age in range(2):
#         dat = dat_set[age][tst]
#         sim = sat_eta(dat, T_parts=sh.T_hl, T_ord=phiT.T_ord)
#         plt.plot(dat.ssd['t'], sim.str_frac, label=dat.name)
#
#     plt.legend()
#     plt.grid()
#     plt.xlabel('t [s]')
#     plt.ylabel('Storage Fraction')

C = ['tab:blue', 'tab:green', 'tab:cyan', 'tab:olive']
for tst in [0, 1, 2]:
    plt.figure(2*tst)
    plt.figure(2*tst+1)
    for age in range(2):
        if age == 0:
            for i in range(0, 4):
                tst_n = tst+3*i
                dat = dat_set[age][tst_n]
                sim = sat_eta(dat, T_parts=sh.T_hl, T_ord=phiT.T_ord)
                plt.figure(2 * tst)
                plt.plot(dat.ssd['t'],
                     [(dat.ssd['F'][k] / dat.ssd['u1'][k]) * sim.eta_sim[k] for k in range(len(dat.ssd['t']))],'--', label=dat.name, color=C[i])
                plt.figure(2 * tst + 1)
                plt.plot(dat.ssd['t'], dat.ssd['T'], label="T_" + dat.name)
        else:
            dat = dat_set[age][tst]
            sim = sat_eta(dat, T_parts=sh.T_hl, T_ord=phiT.T_ord)
            plt.figure(2*tst)
            plt.plot(dat.ssd['t'], [ (dat.ssd['F'][k]/dat.ssd['u1'][k])*sim.eta_sim[k] for k in range(len(dat.ssd['t']))], 'tab:red', label=dat.name)
            plt.figure(2*tst+1)
            plt.plot(dat.ssd['t'], dat.ssd['T'], label="T_"+dat.name)

    plt.figure(2*tst)
    plt.legend()
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel(r'$\eta_{sat, normalized} = \frac{F(k)}{u_1(k)} \eta_{sat}(k)$')
    plt.title(r'$\eta_{sat}$'+" normalized w.r.t flow rate and "+ r'$[NO_x]^{in}$')
    plt.savefig("./figs/max_nox_"+dat.name+".png")

    plt.figure(2*tst+1)
    plt.legend()
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel("Temperature "+uc.units['T'])
    plt.savefig("./figs/max_nox_" + dat.name + "_T.png")

plt.show()

