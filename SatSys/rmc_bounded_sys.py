import matplotlib.pyplot as plt
import  matplotlib
matplotlib.use('tkAgg')
import numpy as np
from Regressor import  km_data as  km
from DataProcessing import decimate_data as dd
from scipy.optimize import linprog
from DataProcessing import unit_convs as uc
from temperature import phiT

dat = dd.decimatedTestData(1, 0)

data_len = len(dat.ssd['t'])

ord = 2

# For linear program
# c = sum of all phi^T \theta
# A stacked up phi^T
# b stacked up eta_kp1

# Constructing the matrices for linear programming
A = np.zeros([data_len-1, ord+1])
b = np.zeros(data_len-1)
for k in range(data_len-1):
    # Aged stuff
    u1_k = dat.ssd['u1'][k]
    F_k = dat.ssd['F'][k]
    T_k = dat.ssd['T'][k]
    phi_ag = (u1_k/F_k) * (phiT.phi_T(T_k, ord)).flatten()
    A[k,:] = phi_ag
    b[k] = dat.ssd['eta'][k+1]
c = np.sum(A, axis=0)


# Solving the linear program
sol = linprog(c, -A, -b, bounds=(None, None))
print(sol)
theta = np.matrix(sol.x).T
print(theta)

# Simulation
eta_sim = np.zeros(data_len)
eta_sim[0] = dat.ssd['eta'][0]
for k in range(data_len-1):
    u1_k = dat.ssd['u1'][k]
    F_k = dat.ssd['F'][k]
    T_k = dat.ssd['T'][k]
    T_poly = [T_k ** n for n in range(ord, -1, -1)]
    phi_ag = (u1_k/F_k)* np.matrix(T_poly).T
    eta_sim[k+1] = (phi_ag.T @ theta)[0, 0]

plt.figure()
plt.plot(dat.ssd['t'], dat.ssd['eta'], label="Data")
plt.plot(dat.ssd['t'], eta_sim, label="Saturated NO_x Predicted")
# plt.plot(dat.ssd['t'], dat.ssd['T'], '--', label="Temperature "+uc.units['T'])
plt.plot(dat.ssd['t'], dat.ssd['u2'], '--', label="Urea Dosing "+uc.units['u2'])
plt.plot(dat.ssd['t'], dat.ssd['F'], '--', label="Flow Rate "+uc.units['F'])
plt.grid(True)
plt.legend()
plt.title(dat.name+"_T_"+str(ord))
plt.xlabel('Time [s]')
plt.ylabel(r'$\eta$' + uc.units['eta'])
plt.savefig("./figs/SatSys_"+dat.name+"_T_"+str(ord)+".png")
plt.show()

