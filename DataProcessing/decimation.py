import numpy as np
import scipy.signal as sig


# =========================================================================================
def decimate2OneHz(x):
    """ Decimate a 5Hz signal to 1Hz signal."""
    y = sig.decimate(x, 5, n=7, ftype='iir', zero_phase=True)
    return y

# ==========================================================================================
def decimate_withTD(tskips, x):
    """ Decimate an array with time skips """
    y = []
    for i in range(len(tskips)-1):
        y.append(decimate2OneHz(x[tskips[i]:tskips[i+1]]))
    return np.array(y).flatten()

# ===========================================================================================
def decimate_time2OneHz(tskips, t):
    """ Decimated the time of ssd and iod to 1 Hz with time skips """
    s = []
    q = 5
    for i in range(len(tskips)-1):
        s_des = t[tskips[i]] + np.arange(np.ceil(len(t[tskips[i]:tskips[i+1]])/q))
        s = np.concatenate((s, s_des), axis=0)
    return np.array(s).flatten()

# ============================================================================================

# Test and example
if __name__ == '__main__':
    import filt_data as fl
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use("tkAgg")

    dat = fl.FilteredTestData(1, 2)

    s = dat.iod['y1']
    t = dat.iod['t']
    tskips = dat.iod['t_skips']

    dec_s = decimate_withTD(tskips, s)
    dec_t = decimate_time2OneHz(tskips, t)

    plt.figure()
    plt.plot(t, s, '--x')
    plt.plot(dec_t, dec_s, '--+')
    plt.show()
