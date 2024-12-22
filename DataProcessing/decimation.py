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
        s_des = np.linspace(t[tskips[i]], t[tskips[i+1]-1], int(len(t[tskips[i]:tskips[i+1]-1])/q)+1)
        s.append(s_des)
    return np.array(s).flatten()

# ============================================================================================

# Test and example
if __name__ == '__main__':
    import filt_data as fl
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use("tkAgg")

    dat = fl.FilteredTestData(0, 2)

    s = dat.ssd['x1']
    t = dat.ssd['t']
    tskips = dat.ssd['t_skips']

    dec_s = decimate_withTD(tskips, s)
    dec_t = decimate_time2OneHz(tskips, t)

    plt.figure()
    plt.plot(t, s, '--x')
    plt.plot(dec_t, dec_s, '--+')
    plt.show()
