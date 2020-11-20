import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import matplotlib.ticker as ticker
import filter
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors

def plot_whitenoise(ax):
    dt = 0.001              # integration time step in seconds
    tmax = 1.01             # stimulus duration
    cutoff = 50.0           # highest frequency in stimulus
    stimulus = whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    ax.plot(time, stimulus)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('White noise')
    

def plot_stimulus(ax):
    dt = 0.001              # integration time step in seconds
    tmax = 1.01             # stimulus duration
    cutoff = 50.0           # highest frequency in stimulus
    stimulus = 6.0 + 2.0*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    ax.plot(time, stimulus)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    

def plot_transfer(axf, axfl, axa, axal):
    """ Plot transfer fucntion for spike frequency and adaptation level.
    """
    dt = 0.0001               # integration time step in seconds
    tmax = 100.0              # stimulus duration
    cutoff = 1010.0           # highest frequency in stimulus
    nfft = 2**12              # number of samples for Fourier trafo
    stimulus = 6.0 + 2.0*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    rate, adapt = adaptation(time, stimulus, alpha=0.05, taua=0.02)
    frate = isi_lowpass(time, rate)
    # transfer function stimulus - rate:
    freqs, csd = sig.csd(stimulus, rate, fs=1.0/dt, nperseg=nfft)
    freqs, psd = sig.welch(stimulus, fs=1.0/dt, nperseg=nfft)
    rtransfer = csd/psd
    rgain = np.abs(rtransfer)
    # transfer function stimulus - spike frequency:
    freqs, csd = sig.csd(stimulus, frate, fs=1.0/dt, nperseg=nfft)
    ftransfer = csd/psd
    fgain = np.abs(ftransfer)
    # transfer function stimulus - adaptation:
    freqs, csd = sig.csd(stimulus, adapt, fs=1.0/dt, nperseg=nfft)
    atransfer = csd/psd
    again = np.abs(atransfer)
    # gain is only meaningful up to cutoff frequency:
    axf.plot(freqs[freqs<cutoff], rgain[freqs<cutoff], label=r'$f(t)$')
    axf.plot(freqs[freqs<cutoff], fgain[freqs<cutoff], label=r'$\langle f(t) \rangle $')
    axf.set_ylabel('Rate gain')
    axf.legend()
    axfl.plot(freqs[freqs<cutoff], rgain[freqs<cutoff])
    axfl.plot(freqs[freqs<cutoff], fgain[freqs<cutoff])
    axfl.set_xscale('log')
    axfl.set_yscale('log')
    axfl.set_ylim(1e0, 1e2)
    axa.plot(freqs[freqs<cutoff], again[freqs<cutoff])
    axa.set_xlabel('Frequency [Hz]')
    axa.set_ylabel('Adaptation gain')
    axal.plot(freqs[freqs<cutoff], again[freqs<cutoff])
    axal.set_xscale('log')
    axal.set_yscale('log')
    axal.set_ylim(1e-2, 1e0)
    axal.set_xlabel('Frequency [Hz]')


def filter_demo():
    """ Demo of the filter properties of spike-frequency adaptation.
    """
    plt.rcParams['axes.xmargin'] = 0.0
    fig, axs = plt.subplots(3, 2, constrained_layout=True)
    plot_whitenoise(axs[0,0])
    plot_stimulus(axs[0,1])
    plot_transfer(axs[1,0], axs[1,1], axs[2,0], axs[2,1])
    plt.show()

        
if __name__ == "__main__":
    filter_demo()
    
