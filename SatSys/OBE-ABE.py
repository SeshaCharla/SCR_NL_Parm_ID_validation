import numpy as np

class OBE_ABE:
        """ Adaptive Sub-OBE-ABE algorithm for general data """

        def __init__(self, y:np.ndarray, X:np.ndarray, gamma_0:float, theta_0:np.ndarray, eps:float = 0)->None :
                """ Initiates the algorithm for OBE-ABE algorithm"""
                self.y = y      # Vector
                self.X = X      # Matrix
                self.N, self.m = np.shape(X)
                self.M = 20     # int(np.max([np.sqrt(self.N), self.m*3]))
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
                self.I = 0      # Interval Length
                # Initialization
                self.gamma[0] = gamma_0
                self.theta[0,:] = theta_0.T
                self.P[(self.m*0):(1*self.m), :] = np.eye(self.m) * 1e-6
                # Running the recursion
                for i in range(1, self.N):
                        self.recursion(i)
        # ==============================================================================

        def recursion(self, n:int) -> None:
                """Recursion steps"""
                self.gamma[n] = self.gamma[n-1]
                self.conditions_for_recursion(n)
                if self.c[n] < 0:
                        self.I = 0
                        self.RLS_recursion(n)
                else:
                        self.I += 1
                        self.no_rls_update(n)
                        if self.I >= self.M:
                                self.update_bounds(n)
                        else:
                                pass
        # =================================================================================

        def update_bounds(self, n:int) -> None:
                """ Update bounds """
                J = (np.argmax(np.square(self.epsilon[n-self.M:n])))+(n - self.M)
                d_J = ( (self.kappa[J-1] * self.G[J])/self.m
                                - (self.eps * (2*np.sqrt(self.gamma[n-1]) - self.eps)) )
                print(d_J)
                if d_J > 0:
                        self.gamma[n] = self.gamma[n-1] - d_J
                else:
                        self.gamma[n] = self.gamma[n-1]
        # =====================================================================================


        def calc_abc(self, n):
                """ Update values for a, b, c after gamma update """
                if n >= 1:
                        self.calc_a(n)
                        self.calc_b(n)
                        self.calc_c(n)
                else:
                        pass
        # ===============================================================

        def conditions_for_recursion(self, n:int) -> None:
                """ before the decision of going into RLS update """
                if n >= 1:
                        self.calc_G(n)
                        self.calc_epsilon(n)
                        self.calc_abc(n)
                else:
                        pass
        # =============================================================

        def RLS_recursion(self, n:int) -> None:
                """ The recursive least squares like recursion part of the OBE-ABE algorithm """
                if n >= 1:
                        self.calc_lambda(n)
                        self.calc_alpha(n)
                        self.calc_beta(n)
                        self.calc_P(n)
                        self.calc_theta(n)
                        self.calc_kappa(n)
                else:
                        pass
        # =============================================================================================

        def no_rls_update(self, n:int) -> None:
                """ No update on the ellipsoid """
                if n>=1:
                        self.lmbda[n] = self.lmbda[n-1]
                        self.alpha[n] = self.alpha[n-1]
                        self.beta[n] = self.beta[n-1]
                        self.P[(n*self.m):((n+1)*self.m), :] = self.P[((n-1)*self.m):((n)*self.m), :]
                        self.theta[n, :] = self.theta[n-1,:]
                        self.kappa[n] = self.kappa[n-1]
                else:
                        pass
        # =========================================================================================

        def calc_G(self, n:int) -> None:
                """ Calculate G(n) """
                if (n>=1):
                        x_n = np.matrix(self.X[n, :]).T
                        P_nm1 = np.matrix(self.P[(n-1)*self.m:(n)*self.m, :])
                        self.G[n] = (x_n.T @ P_nm1 @ x_n)[0, 0]
                else:
                        self.G[n] = 0
        # ==========================================================

        def calc_epsilon(self, n:int) -> None:
                """ epsilon(n) """
                if (n>=1):
                        x_n = np.matrix(self.X[n,:]).T
                        theta_n = np.matrix(self.theta[n,:]).T
                        y_hat_n = theta_n.T @ x_n
                        self.epsilon[n] = y[n] - y_hat_n[0, 0]
                else:
                        self.epsilon[n] = 0
        # ======================================================

        def calc_a(self, n:int) -> None :
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

        def calc_b(self, n:int) -> None :
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

        def calc_c(self, n:int) -> None :
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

        def calc_lambda(self, n:int) -> None :
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

        def calc_alpha(self, n:int) -> None :
                """ alpha (n)"""
                self.alpha[n] = 1 - self.lmbda[n]
        # =========================================

        def calc_beta(self, n:int) -> None :
                """ beta(n) """
                self.beta[n] = self.lmbda[n]
        # ==================================================

        def calc_P(self, n:int ) -> None :
                """ Calculating P, Covariance matrix """
                if n>=1:
                        alpha_n = self.alpha[n]
                        beta_n = self.beta[n]
                        P_nm1 = np.matrix(self.P[((n-1)*self.m):(n*self.m), :])
                        x_n = np.matrix(self.X[n, :]).T
                        G_n = self.G[n]
                        self.P[((n)*self.m):((n+1)*self.m), :] = (1/alpha_n)*(P_nm1 -
                                                                                ( (beta_n)/(alpha_n + beta_n*G_n)
                                                                                        * (P_nm1@x_n@x_n.T@P_nm1) ) )
                else:
                        pass
        # ==============================================================================================================

        def calc_theta(self, n:int) -> None:
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

        def calc_kappa(self, n:int) -> None:
                """ kappa[n] """
                if n>=1:
                        alpha_n = self.alpha[n]
                        beta_n = self.beta[n]
                        epsilon_n = self.epsilon[n]
                        G_n = self.G[n]
                        gamma_n = self.gamma[n]
                        self.kappa[n] = ( (alpha_n * self.kappa[n-1]) + beta_n*gamma_n
                                                - (alpha_n*beta_n*epsilon_n**2)/(alpha_n + beta_n*G_n) )
                else:
                        pass
        # ==================================================================================================



if __name__ == "__main__":
        import matplotlib.pyplot as plt
        N = 50
        a1 = 2
        a2 = -1.48
        a3 = 0.34
        y = np.zeros(N)
        v = np.zeros(N)
        y_noise = np.zeros(N)
        phi = np.zeros([N, 3])
        y[0] = 2
        y_noise[0] = 2
        y[1] = 1
        y_noise[1] = 1
        y[2] = 0
        y_noise[2] = 0
        for i in range(3,N):
                v[i] = 0.5*(np.random.rand())-0.5
                y[i] = a1*y[i-1] + a2 * y[i-2] + a3 * y[i-3]
                y_noise[i] = y[i] + v[i]
                phi[i,:] = [y[i-1], y[i-2], y[i-3]]
        obe_alg = OBE_ABE(y[3:], phi[3:, :], theta_0= np.matrix([[0], [0], [0]]), eps=0,  gamma_0=1)
        plt.figure()
        plt.plot(y, label='y')
        plt.plot(y_noise, label="y-noise")
        plt.legend()
        plt.grid()
        plt.show()
