import numpy as np

#=======================================================================================================================
def cdRLS_smooth(y, lmda=0, nu=0, h=0):
    """yt: time series data
       lmda: forgetting factor
       nu: drift parameter
       h: threshold parameter
       return: smoothed data
       Added a median filter with window lenth med_win=5 for removing one off spikes
    """
    n = len(y)
    med_win = 5         ## window of the median filter
    # initialize variables
    th_hat = np.zeros(n)
    eps    = np.zeros(n)
    s1     = np.zeros(n)
    s2     = np.zeros(n)
    g1     = np.zeros(n)
    g2     = np.zeros(n)
    th_hat[0] = y[0]
    l = 0
    for t in range(1,n):
        # update variables
        l = l+1
        eps[t] = y[t] - th_hat[t-1]
        s1[t] = eps[t]
        s2[t] = -eps[t]
        g1[t] = np.max([g1[t-1] + s1[t] - nu, 0])
        g2[t] = np.max([g2[t-1] + s2[t] - nu, 0])
        if (g1[t] > h) or (g2[t] > h):  # change detection
            th_hat[t] = y[t]            # reset rls estimate
            g1[t] = 0
            g2[t] = 0
            l = 0
        else:
            th_hat[t] = lmda * th_hat[t - 1] + (1 - lmda) * y[t]        # RLS estimate
            if l > med_win:
                th_hat[t] = np.median(th_hat[(t-med_win):t], axis=0)  # Median filter after RLS
    return th_hat, g1, g2


# ======================================================================================================================
def cdRLS_withTD(t_skips, y, lmda=0, nu=0, h=0):
    """CD-RLS on data with time-discontinuities
        yt: time series data
       lmda: forgetting factor
       nu: drift parameter
       h: threshold parameter
       return: smoothed data
    """
    n = len(y)
    # initialize variables
    th_hat = np.zeros(n)
    g1     = np.zeros(n)
    g2     = np.zeros(n)
    for i in range(0,len(t_skips)-1):
        th, g_1, g_2 = cdRLS_smooth(y[t_skips[i]:t_skips[i+1]], lmda, nu, h)
        th_hat[t_skips[i]:t_skips[i+1]] = th
        g1[t_skips[i]:t_skips[i+1]] = g_1
        g2[t_skips[i]:t_skips[i+1]] = g_2
    return th_hat, g1, g2


# ======================================================================================================================
class cdRLS_parms:
    def __init__(self, str):
        if str == "test":
            self.lmbda = 0.98
            self.h = dict()
            self.nu = dict()
            # Values
            self.nu['x1'] = 0.5
            self.nu['x2'] = 0.04
            self.nu['u1'] = 0.5
            self.nu['u2'] = 0.1
            self.nu['T'] = 30
            self.nu['F'] = 20
            self.nu['y1'] = 0.5
            #
            self.h['x1'] = 1
            self.h['x2'] = 0.04
            self.h['u1'] = 1
            self.h['u2'] = 0.2
            self.h['T'] = 40
            self.h['F'] = 50
            self.h['y1'] = 1

        elif str == "truck":
            self.lmbda = 0.95
            self.h = dict()
            self.nu = dict()
            # Values
            self.nu['y1'] = 2.5
            self.nu['u1'] = 20
            self.nu['u2'] = 1
            self.nu['T'] = 30
            self.nu['F'] = 200
            #
            self.h['y1'] = 5
            self.h['u1'] = 40
            self.h['u2'] = 2
            self.h['T'] = 40
            self.h['F'] = 400
        else:
            raise(ValueError("Wrong string argument"))


# ======================================================================================================================