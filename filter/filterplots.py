import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import matplotlib.ticker as ticker
import filter
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors


def plot_whitenoise():
    dt = 0.001              # integration time step in seconds
    tmax = 1.0              # stimulus duration
    cutoff = 100.0          # highest frequency in stimulus
    rng = np.random.RandomState(283)
    stimulus = filter.whitenoise(0.0, cutoff, dt, tmax, rng)
    time = np.arange(len(stimulus))*dt
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.axhline(0.0, linestyle='--', color=colors['gray'], lw=0.5)
    ax.axhline(-1.0, linestyle=':', color=colors['gray'], lw=0.5)
    ax.axhline(+1.0, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(time, stimulus)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('White noise')
    fig.savefig('filter-whitenoise')
    

def plot_stimulus():
    dt = 0.001              # integration time step in seconds
    tmax = 1.0              # stimulus duration
    cutoff = 50.0           # highest frequency in stimulus
    mean = 5.0              # stimulus mean
    stdev = 2.5             # stimulus standard deviation
    rng = np.random.RandomState(981)
    stimulus = mean + stdev*filter.whitenoise(0.0, cutoff, dt, tmax, rng)
    time = np.arange(len(stimulus))*dt
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.axhline(mean, linestyle='--', color=colors['gray'], lw=0.5)
    ax.axhline(mean-stdev, linestyle=':', color=colors['gray'], lw=0.5)
    ax.axhline(mean+stdev, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(time, stimulus)
    ax.set_ylim(-2, 12)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    fig.savefig('filter-stimulus')
    

def plot_stimulus_psd():
    dt = 0.001              # integration time step in seconds
    tmax = 100.0            # stimulus duration
    cutoff = 200.0          # highest frequency in stimulus
    mean = 5.0              # stimulus mean
    stdev = 2.5             # stimulus standard deviation
    nfft = 2**10            # nfft for power spectrum estimate
    rng = np.random.RandomState(981)
    stimulus = mean + stdev*filter.whitenoise(0.0, cutoff, dt, tmax, rng)
    freqs, psd = sig.welch(stimulus, fs=1.0/dt, nperseg=nfft)
    fig, axs = plt.subplots(1, 2, figsize=(figwidth, 0.4*figwidth))
    ax = axs[0]
    ax.plot(freqs, psd)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power')
    ax = axs[1]
    ax.plot(freqs, 10.0*np.log10(psd/np.max(psd)))
    ax.set_ylim(-20, 0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    fig.savefig('filter-psd')
    

def plot_rate_psd():
    dt = 0.001              # integration time step in seconds
    tmax = 100.0            # stimulus duration
    cutoff = 200.0          # highest frequency in stimulus
    mean = 5.0              # stimulus mean
    stdev = 2.5             # stimulus standard deviation
    nfft = 2**10            # nfft for power spectrum estimate
    rng = np.random.RandomState(981)
    stimulus = mean + stdev*filter.whitenoise(0.0, cutoff, dt, tmax, rng)
    time = np.arange(len(stimulus))*dt
    rate, adapt = filter.adaptation(time, stimulus, alpha=0.05, taua=0.02)
    freqs, psd = sig.welch(rate, fs=1.0/dt, nperseg=nfft)
    fig, axs = plt.subplots(1, 2, figsize=(figwidth, 0.4*figwidth))
    ax = axs[0]
    ax.plot(freqs, psd)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power')
    ax = axs[1]
    ax.plot(freqs, 10.0*np.log10(psd/np.max(psd)))
    ax.set_ylim(-20, 0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    fig.savefig('filter-ratepsd')
    

def plot_transfer(axf, axfl, axa, axal):
    """ Plot transfer fucntion for spike frequency and adaptation level.
    """
    dt = 0.0001               # integration time step in seconds
    tmax = 100.0              # stimulus duration
    cutoff = 1010.0           # highest frequency in stimulus
    nfft = 2**12              # number of samples for Fourier trafo
    stimulus = 6.0 + 2.0*filter.whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    rate, adapt = filter.adaptation(time, stimulus, alpha=0.05, taua=0.02)
    frate = filter.isi_lowpass(time, rate)
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

        
if __name__ == "__main__":
    plot_whitenoise()
    plot_stimulus()
    plot_stimulus_psd()
    plot_rate_psd()


    
