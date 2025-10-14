import numpy as np

class OBE_ABE:
        """ Adaptive Sub-OBE-ABE algorithm for general data """

        def __init__(self, y, X, eps, gamma_0, theta_0, M_frac):
                """ Initiates the algorithm for OBE-ABE algorithm"""
                self.y = y
                self.X = X
                self.N, self.m = np.shape(X)
                self.M = int(self.N*M_frac)
                self.eps = eps
                self.gamma = np.zeros(self.N)
                self.gamma[0] = gamma_0
                self.theta = np.zeros([self.N, self.m])
                self.theta[0,:] = theta_0.T
                # List of all the variables
                self.P = np.zeros([self.N*self.m, self.m])
                self.G = np.zeros([self.N*self.m, self.m])
                self.epsilon = np.zeros(self.N)         # Innovation
                self.kappa = np.zeros(self.N)           # Kappa
                self.a = np.zeros(self.N)
                self.b = np.zeros(self.N)
                self.c = np.zeros(self.N)
                self.lmbda = np.zeros(self.N)
                self.alpha = np.zeros(self.N)
                self.beta = np.zeros(self.N)

        def calc_G_n(self, n):
                """ Calculate G(n) """
                if (n>=1):
                        x_n = self.X[n].T
                        P_nm1 = self.P[(n-1)*self.m:(n)*self.m, :]
                        self.G[n:n+self.m, :] = x_n.T @ P_nm1 @ x_n
                else:
                        self.G[n*self.m:(n+1)*self.m, :] = np.zeros([self.m, self.m])

        def calc_epsilon_n(self, n):
                """ epsilon(n) """
                if (n>=1):
                        x_n = self.X[n].T
                        theta_n = np.matrix(self.theta[n,:]).T
                        y_hat_n = theta_n.T @ x_n
                        self.epsilon[n] = y[n] - y_hat_n[0, 0]
                else:
                        self.epsilon[n] = 0


        def calc_P_n(self, n):
                """ Calculate P[n] """
                if (n>=1):
                        P_nm1 = self.P[(n-1)*self.m:n*self.m, :]

        def a_n(self, n):
                """ Calculate a_n """
