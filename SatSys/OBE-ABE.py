import numpy as np

class OBE_ABE:
        """ Adaptive Sub-OBE-ABE algorithm for general data """

        def __init__(self, y, X, eps, gamma_0, theta_0, M_frac):
                """ Initiates the algorithm for OBE-ABE algorithm"""
                self.y = y      # Vector
                self.X = X      # Matrix
                self.N, self.m = np.shape(X)
                self.M = int(self.N*M_frac)
                self.eps = eps
                # List of all arrays storing recursing varables
                # Matrix/Vector arrays
                self.theta = np.zeros([self.N, self.m])
                self.P = np.zeros([self.N*self.m, self.m])
                # Scalar arrays
                self.gamma = np.zeros(self.N)
                self.G = np.zeros(self.N)
                self.epsilon = np.zeros(self.N)         # Innovation
                self.kappa = np.zeros(self.N)           # Kappa
                self.a = np.zeros(self.N)
                self.b = np.zeros(self.N)
                self.c = np.zeros(self.N)
                self.d = np.zeros(self.N)
                self.lmbda = np.zeros(self.N)
                self.alpha = np.zeros(self.N)
                self.beta = np.zeros(self.N)
                # Initialization
                self.gamma[0] = gamma_0
                self.theta[0,:] = theta_0.T
                self.P[(self.m*0):(1*self.m), :] = np.eye(3) * 1e-6
        # ==============================================================================

        def calc_G(self, n):
                """ Calculate G(n) """
                if (n>=1):
                        x_n = self.X[n, :].T
                        P_nm1 = self.P[(n-1)*self.m:(n)*self.m, :]
                        self.G[n] = x_n.T @ P_nm1 @ x_n
                else:
                        self.G[n] = 0
        # ==========================================================

        def calc_epsilon(self, n):
                """ epsilon(n) """
                if (n>=1):
                        x_n = self.X[n,:].T
                        theta_n = np.matrix(self.theta[n,:]).T
                        y_hat_n = theta_n.T @ x_n
                        self.epsilon[n] = y[n] - y_hat_n[0, 0]
                else:
                        self.epsilon[n] = 0
        # ======================================================

        def calc_a(self, n):
                """ a(n) """
                if n>=1:
                        m = self.m
                        gamma_n = self.gamma[n]
                        epsilon_n = self.epsilon[n]
                        G_n = self.G[n]
                        kappa_nm1 = self.kappa[n-1]
                        self.a[n] = ( (m * gamma_n) - (m * epsilon_n**2) + (m * G_n**2 * gamma_n) - (2*m*G_n*gamma_n)
                                        - (kappa_nm1 * G_n) + (kappa_nm1 * G_n**2) + (G_n * gamma_n)
                                        - (G_n**2 * gamma_n) - (epsilon_n**2 * G_n) )
                else:
                        self.a[n] = 0
        # ==============================================================================================================

        def calc_b(self, n):
                """ b(n) """
                if n>=0:
                        m = self.m
                        gamma_n = self.gamma[n]
                        epsilon_n = self.epsilon[n]
                        G_n = self.G[n]
                        kappa_nm1 = self.kappa[n-1]
                        self.b[n] = ( (2*m * epsilon_n**2) - (2*m*gamma_n) + (2*m * G_n * gamma_n)
                                        + (2 * kappa_nm1 * G_n) - (kappa_nm1 * G_n**2) - (G_n * gamma_n)
                                        + (epsilon_n**2 * G_n) )
                else:
                        self.b[n] = 0
        # ======================================================================================================

        def calc_c(self, n):
                """ c(n) """
                if n>=0:
                        m = self.m
                        gamma_n = self.gamma[n]
                        epsilon_n = self.epsilon[n]
                        G_n = self.G[n]
                        kappa_nm1 = self.kappa[n-1]
                        self.c[n] = ( (m * gamma_n) - (m * epsilon_n**2) - (kappa_nm1 * G_n))
                else:
                        self.c[n] = 0
        # ========================================================================================

        def calc_lambda(self, n):
                """ lambda(n) """
                if n>=0:
                        an = self.a[n]
                        bn = self.b[n]
                        cn = self.c[n]
                        if cn < 0:
                                self.lmbda[n] = (-bn + np.sqrt( bn**2 - 4*an*cn))/(2*an)
                        else:
                                self.lmbda[n] = 0
                else:
                        self.lmbda[n] = 0
        # ====================================================================================

        def calc_alpha(self, n):
                """ alpha (n)"""
                self.alpha[n] = 1 - self.lmbda[n]
        # =========================================

        def calc_beta(self, n):
                """ beta(n) """
                self.beta[n] = self.lmbda[n]
        # ==================================================

        def calc_P(self, n):
                """ Calculating P, Covariance matrix """
                if n>=1:
                        alpha_n = self.alpha[n]
                        beta_n = self.beta[n]
                        P_nm1 = self.P[((n-1)*self.m):(n*self.m), :]
                        x_n = np.matrix(self.X[n, :]).T
                        G_n = self.G[n]
                        self.P[((n)*self.m):((n+1)*self.m), :] = (1/alpha_n)*(P_nm1 -
                                                                                ( (beta_n)/(alpha_n + beta_n*G_n)
                                                                                        * (P_nm1@x_n@x_n.T@P_nm1) ) )
                else:
                        pass
        # ==============================================================================================================

        def calc_theta(self, n):
                """ Calculating the center theta """
                if n>=1:
                        theta_nm1 = np.matrix(self.theta[n-1, :]).T
                        beta_n = self.beta[n]
                        x_n = np.matrix(self.X[n, :]).T
                        epsilon_n = self.epsilon[n]
                        P_n = self.P[((n)*self.m):((n+1)*self.m), :]
                        theta_n = theta_nm1 + beta_n * (P_n @ x_n) * epsilon_n
                        self.theta[n,:] = theta_n.T
                else:
                        pass
        # ===========================================================================

        def calc_kappa(self, n):
                """ kappa[n] """
                if n>=1:
                        alpha_n = self.alpha[n]
                        beta_n = self.beta[n]
                        epsilon_n = self.epsilon[n]
                        G_n = self.G[n]



if __name__ == "__main__":
        import matplotlib.pyplot as plt

        a1 = 2
        a2 = -1.48
        a3 = 0.34
        y = np.zeros(100)
        v = np.zeros(100)
        y_noise = np.zeros(100)
        phi = np.zeros([100, 3])
        y[0] = 1
        y_noise[0] = 1
        y[1] = 2
        y_noise[1] = 2
        y[2] = 3
        y_noise[2] = 3
        for i in range(3,100):
                v[i] = 2*(np.random.rand())-1
                y[i] = a1*y[i-1] + a2 * y[i-2] + a3 * y[i-3]
                y_noise[i] = y[i] + v[i]
                phi[i,:] = [y[i-1], y[i-2], y[i-3]]
        obe_alg = OBE_ABE(y[3:], phi[3:, :], theta_0= np.matrix([[0], [0], [0]]), eps=1e-3, gamma_0=1.5, M_frac=0.2)
        plt.figure()
        plt.plot(y, label='y')
        plt.plot(y_noise, label="y-noise")
        plt.legend()
        plt.grid()
        plt.show()
