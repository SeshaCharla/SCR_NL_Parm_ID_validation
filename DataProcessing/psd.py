import numpy as np
from scipy.signal import welch

def welch_psd(y, fs):
    """Wrapper around scipy.signal.welch
    With default parameters
    """
    seg_length = 2**8
    nf = 2**4
    f, pd = welch(y,
                  fs=fs,
                  window='bartlett',
                  nperseg=seg_length,
                  noverlap=None,
                  nfft=nf*seg_length,
                  detrend='linear',
                  return_onesided=True,
                  scaling='spectrum',
                  axis=-1,
                  average='mean')
    return f, pd

# ===============================================================================================================
def welch_TD(y, t_skips, fs):
    """ Welch PSD with time-discontinuities
    """
    n = len(t_skips)-1
    psd = list()
    f = list()
    for i in range(n):
        if len(y[t_skips[i]:t_skips[i+1]]) > 0:
            f_i, psd_i = welch_psd(y[t_skips[i]:t_skips[i+1]], fs)
            f.append(f_i)
            psd.append(psd_i)
    f = np.array(f).flatten()
    psd = np.array(psd).flatten()
    return f, psd/np.max(psd)

# ================================================================================================================
