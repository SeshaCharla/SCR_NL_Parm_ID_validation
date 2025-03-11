import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
#
from DataProcessing import decimate_data as dd
from DataProcessing import unit_convs as uc
import switching_handler as sh

dct = dd.load_decimated_test_data_set()
fig_dpi = 300
key = 'T'

lines = (sh.switch_handle(sh.T_hl)).T_parts

# Plotting all the Data sets
plt.figure()
for i in range(2):
    for j in range(3):
        plt.plot(dct[i][j].ssd['t'], dct[i][j].ssd[key], label= dct[i][j].name, linewidth=1)
for line in lines:
    plt.plot(dct[i][j].ssd['t'], line * np.ones(np.shape(dct[i][j].ssd['t'])), 'k--', linewidth=1)
    plt.text(1300, line+0.2, str((line*10)+200) + r'$\, ^0 C$')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel(key + uc.units[key])
plt.title("Temperature plots of Test Cell Data")
plt.savefig("figs/" + "hybrid_ssd_hl_" + key + ".png", dpi=fig_dpi)
plt.show()