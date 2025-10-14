import matplotlib.pyplot as plt
import numpy as np
from Regressor import  km_data as  km
from DataProcessing import decimate_data as dd
from scipy.optimize import linprog
from DataProcessing import unit_convs as uc
from temperature import phiT
import cvxpy as cvx
import pickle

dat = dd.decimatedTestData(1, 2)

data_len = len(dat.iod['t'])

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
    u1_k = dat.iod['u1'][k]
    F_k = dat.iod['F'][k]
    T_k = dat.iod['T'][k]
    phi_ag = (u1_k/F_k) * (phiT.phi_T(T_k, ord)).flatten()
    A[k,:] = phi_ag
    b[k] = dat.iod['eta'][k+1]
c = np.sum(A, axis=0)


# Solving the linear program
# sol = linprog(c, -A, -b, bounds=(None, None))
# print(sol)
# theta = np.matrix(sol.x).T
# print(theta)

# CVX problem
theta_var = cvx.Variable(ord+1)
gamma = cvx.Variable(data_len-1)
prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(A @ theta_var - b) + cvx.sum_squares(gamma)),
                   [A @ theta_var - b >=  -gamma,
                    gamma >= 0,
                    gamma <= np.max(np.abs(dat.ssd['eta']-dat.iod['eta']))])
prob.solve()
theta = theta_var.value

# Simulation
eta_sim = np.zeros(data_len)
eta_sim[0] = dat.iod['eta'][0]
age_sig = np.zeros(data_len-1)
for k in range(data_len-1):
    u1_k = dat.iod['u1'][k]
    F_k = dat.iod['F'][k]
    T_k = dat.iod['T'][k]
    T_poly = [T_k ** n for n in range(ord, -1, -1)]
    phi_ag = (u1_k/F_k)* np.matrix(T_poly).T
    eta_sim[k+1] = (phi_ag.T @ theta)[0, 0]
    age_sig[k] = (np.matrix(T_poly) @ theta)[0,0]

plt.figure()
plt.plot(dat.iod['t'], dat.iod['eta'], label="Data")
plt.plot(dat.iod['t'], eta_sim, label="Saturated NO_x Predicted")
# plt.plot(dat.ssd['t'], dat.ssd['T'], '--', label="Temperature "+uc.units['T'])
# plt.plot(dat.ssd['t'], dat.ssd['eta'], '--', label="Urea Dosing "+uc.units['u2'])
plt.plot(dat.iod['t'], dat.iod['F'], '--', label="Flow Rate "+uc.units['F'])
plt.grid(True)
plt.legend()
plt.title(dat.name)
plt.xlabel('Time [s]')
plt.ylabel(r'$\eta$' + uc.units['eta'])
plt.savefig("./SatSys/figs/SatSys_iod_"+dat.name+".png")
plt.show()


plt.figure()
plt.plot(dat.iod['t'][1:], gamma.value, label=r'$\gamma(k)$')
plt.plot(dat.iod['t'], dat.iod['y1'] - np.mean(dat.iod['y1'][1:20]-dat.ssd['x1'][1:20]), label="NOx Sensor Data")
plt.plot(dat.ssd['t'], dat.ssd['x1'], label="FTIR NOX")
plt.plot(dat.ssd['t'], (dat.ssd['x2'] - np.min(dat.ssd['x2'])), label="FTIR NH3")
# plt.plot(dat.ssd['t'], dat.ssd['T']/10, label="T/10"+uc.units['T'])
plt.xlabel('Time [s]')
plt.ylabel(r'Concentration of Species' + uc.units['eta'])
plt.legend()
plt.title(dat.name+" cross sensitivity error")
plt.grid()
plt.savefig("./SatSys/figs/SatSys_iod_gamma_"+dat.name+".png")
plt.show()

data = dict()
data['t'] = dat.iod['t']
data['age_sig'] = age_sig

with open(dat.name+'-age_sig.pkl', 'wb') as file:
    pickle.dump(data, file)
