import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')
#
from DataProcessing import decimate_data as dd
from DataProcessing import unit_convs as uc

dct = dd.load_decimated_test_data_set()
tsts = [12, 15]
fig_dpi = 600

# Plotting all the Data sets
plt.figure()
for i in range(2):
    for j in range(tsts[i]):
        eta = dct[i][j].ssd['eta']
        u1 = dct[i][j].ssd['u1']
        nu = np.array([max(0,eta[i]/u1[i-1]) for i in range(1, len(u1))])
        nu = np.concatenate(([nu[0]], nu));
        mu = dct[i][j].ssd['mu']
        plt.scatter(mu, nu, marker='x', label= dct[i][j].name)
plt.grid()
plt.legend()
plt.xlabel('ANR')
plt.ylabel("NOx Reduction Efficiency")
plt.title("ANR vx NOx Reduction Efficiency")
plt.savefig("figs/" + "ANR-NOx.png", dpi=fig_dpi)
plt.show()