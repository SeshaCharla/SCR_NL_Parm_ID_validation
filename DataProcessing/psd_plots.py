import numpy as np
import rdRawDat as rd
import filt_data as fd
import decimate_data as dd
import psd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkAgg')

# ===============================================================================================
raw_dat = rd.load_test_data_set()
filt_dat = fd.load_filtered_test_data_set()
dec_dat = dd.load_decimated_test_data_set()



states = ['x1', 'x2', 'u1', 'u2', 'T', 'F', 'eta', 'y1']
dat = dec_dat
for age in range(2):
    for test in range(3):
        for state in states:
            plt.figure(state)
            if state != 'y1':
                time_series = dat[age][test].ssd[state]
                tskips = dat[age][test].ssd['t_skips']
            elif state == 'y1':
                time_series = dat[age][test].iod[state]
                tskips = dat[age][test].iod['t_skips']
            fs = 1 / dat[age][test].dt
            f, mag = psd.welch_TD(time_series, tskips, fs)
            plt.plot(f, mag, label=dat[age][test].name, linewidth=1)
            plt.plot(0.1 * np.ones(np.size(f)), np.linspace(0, 1, np.size(f)), 'k-.', linewidth=1.5)
            plt.xlim([-0.01, 0.2])
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude [normalized]')
            plt.grid(True)
            plt.legend(loc='best')

plt.show()
