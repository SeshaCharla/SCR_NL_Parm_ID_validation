import numpy as np
import  scipy.signal as sig

lp_filt = sig.butter(13, 0.1, 'low', analog=False, fs=5, output='sos')

def sosff_TD(tskips, x):
    """Filter the data with time jumps"""
    n = len(x)
    y = np.zeros(n)
    for i in range(len(tskips)-1):
        if len(x[tskips[i-1]:tskips[i]]) > 2*13:
            y[tskips[i-1]:tskips[i]] =




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    w, h = sig.sosfreqz(lp_filt, worN=1500)

    plt.subplot(2, 1, 1)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(w/np.pi, db)
    plt.ylim(-75, 5)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    plt.subplot(2, 1, 2)
    plt.plot(w/np.pi, np.angle(h))
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    plt.show()