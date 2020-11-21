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
    ax.plot(time, stimulus, color=colors['green'])
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('White noise')
    fig.savefig('filter-whitenoise')
    

def plot_stimulus():
    dt = 0.001              # integration time step in seconds
    tmax = 1.0              # stimulus duration
    cutoff = 50.0           # highest frequency in stimulus
    mean = 5.0              # stimulus mean
    stdev = 1.5             # stimulus standard deviation
    rng = np.random.RandomState(981)
    stimulus = mean + stdev*filter.whitenoise(0.0, cutoff, dt, tmax, rng)
    time = np.arange(len(stimulus))*dt
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.axhline(mean, linestyle='--', color=colors['gray'], lw=0.5)
    ax.axhline(mean-stdev, linestyle=':', color=colors['gray'], lw=0.5)
    ax.axhline(mean+stdev, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(time, stimulus, color=colors['green'])
    ax.set_ylim(0.0, 10.0)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    fig.savefig('filter-stimulus')
    

def plot_stimulus_psd():
    dt = 0.001              # integration time step in seconds
    tmax = 100.0            # stimulus duration
    cutoff = 200.0          # highest frequency in stimulus
    mean = 5.0              # stimulus mean
    stdev = 1.5             # stimulus standard deviation
    nfft = 2**9             # nfft for power spectrum estimate
    rng = np.random.RandomState(981)
    stimulus = mean + stdev*filter.whitenoise(0.0, cutoff, dt, tmax, rng)
    freqs, psd = sig.welch(stimulus, fs=1.0/dt, nperseg=nfft)
    fig, axs = plt.subplots(1, 2, figsize=(figwidth, 0.4*figwidth))
    ax = axs[0]
    ax.plot(freqs, psd, color=colors['green'])
    ax.set_ylim(0, 0.015)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power')
    ax = axs[1]
    ax.plot(freqs, 10.0*np.log10(psd/np.max(psd)), color=colors['green'])
    ax.set_ylim(-20, 0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    fig.savefig('filter-psd')
    

def plot_rate_psd():
    dt = 0.001              # integration time step in seconds
    tmax = 100.0            # stimulus duration
    cutoff = 200.0          # highest frequency in stimulus
    mean = 5.0              # stimulus mean
    stdev = 1.5             # stimulus standard deviation
    nfft = 2**9             # nfft for power spectrum estimate
    rng = np.random.RandomState(781)
    stimulus = mean + stdev*filter.whitenoise(0.0, cutoff, dt, tmax, rng)
    time = np.arange(len(stimulus))*dt
    rate, adapt = filter.adaptation(time, stimulus, alpha=0.05, taua=0.02)
    freqs, psd = sig.welch(rate, fs=1.0/dt, nperseg=nfft)
    fig, axs = plt.subplots(1, 2, figsize=(figwidth, 0.4*figwidth))
    ax = axs[0]
    ax.plot(freqs, psd)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power')
    ax.set_ylim(0, 30)
    ax = axs[1]
    ax.plot(freqs, 10.0*np.log10(psd/np.max(psd)))
    ax.set_ylim(-20, 0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    fig.savefig('filter-ratepsd')
    

def plot_ratetransfer():
    """ Plot transfer fucntion for spike frequency.
    """
    dt = 0.00001              # integration time step in seconds
    tmax = 200.0              # stimulus duration
    cutoff = 1010.0           # highest frequency in stimulus
    nfft = 2**16              # number of samples for Fourier trafo
    stimulus = 5.0 + 1.5*filter.whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    rate, adapt = filter.adaptation(time, stimulus, alpha=0.05, taua=0.02)
    freqs, rgain, rphase = filter.transfer(stimulus, rate, dt, nfft, cutoff)
    # gain plots:
    fig, axs = plt.subplots(1, 2, figsize=(figwidth, 0.4*figwidth))
    ax = axs[0]
    ax.plot(freqs, rgain, color=colors['orange'])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Gain')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 50)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax = axs[1]
    ax.plot(freqs, rgain, color=colors['orange'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(10, 50)
    ax.set_xlabel('Frequency [Hz]')
    fig.savefig('filter-rategain')
    # phase plots:
    fig, axs = plt.subplots(1, 2, figsize=(figwidth, 0.4*figwidth))
    ax = axs[0]
    ax.axhline(0.0, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(freqs, rphase, color=colors['orange'])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase')
    ax.set_xlim(0, 200)
    ax.set_ylim(0.0, 0.25*np.pi)
    ax.set_yticks([0.0, 0.25*np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/4$'])
    ax = axs[1]
    ax.axhline(0.0, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(freqs, rphase, color=colors['orange'])
    ax.set_xscale('log')
    ax.set_ylim(0.0, 0.25*np.pi)
    ax.set_yticks([0.0, 0.25*np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/4$'])
    ax.set_xlabel('Frequency [Hz]')
    fig.savefig('filter-ratephase')
    
    # transfer stimulus - adaptation:
    freqs, again, aphase = filter.transfer(stimulus, adapt, dt, nfft, cutoff)
    fig, axs = plt.subplots(2, 2, figsize=(figwidth, 0.5*figwidth))
    ax = axs[0,0]
    ax.plot(freqs, again, color=colors['blue'])
    ax.set_ylabel('Gain')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.8)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax = axs[0,1]
    ax.plot(freqs, again, color=colors['blue'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e0)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax = axs[1,0]
    ax.axhline(0.0, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(freqs, aphase, color=colors['blue'])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase')
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.5*np.pi, 0.0)
    ax.set_yticks([0.0, -0.25*np.pi, -0.5*np.pi])
    ax.set_yticklabels([r'$0$', r'$-\pi/4$', r'$-\pi/2$'])
    ax = axs[1,1]
    ax.axhline(0.0, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(freqs, aphase, color=colors['blue'])
    ax.set_xscale('log')
    ax.set_ylim(-0.5*np.pi, 0.0)
    ax.set_yticks([0.0, -0.25*np.pi, -0.5*np.pi])
    ax.set_yticklabels([r'$0$', r'$-\pi/4$', r'$-\pi/2$'])
    ax.set_xlabel('Frequency [Hz]')
    fig.savefig('filter-adaptbode')

    # transfer stimulus - spike frequency:
    frate = filter.isi_lowpass(time, rate)
    freqs, fgain, fphase = filter.transfer(stimulus, frate, dt, nfft, cutoff)
    fphase[np.abs(fphase) > 0.25*np.pi] = np.nan
    fig, axs = plt.subplots(2, 2, figsize=(figwidth, 0.5*figwidth))
    ax = axs[0,0]
    ax.plot(freqs, rgain, color=colors['orange'], label=r'$f(t)$')
    ax.plot(freqs, fgain, color=colors['red'], label=r'$\langle f(t) \rangle_T$')
    ax.set_ylabel('Gain')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 50.0)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.legend()
    ax = axs[0,1]
    ax.plot(freqs, rgain, color=colors['orange'])
    ax.plot(freqs, fgain, color=colors['red'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1, 50)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax = axs[1,0]
    ax.axhline(0.0, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(freqs, rphase, color=colors['orange'])
    ax.plot(freqs, fphase, color=colors['red'])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase')
    ax.set_xlim(0, 200)
    ax.set_ylim(0.0, 0.25*np.pi)
    ax.set_yticks([0.0, 0.25*np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/4$'])
    ax = axs[1,1]
    ax.axhline(0.0, linestyle=':', color=colors['gray'], lw=0.5)
    ax.plot(freqs, rphase, color=colors['orange'])
    ax.plot(freqs, fphase, color=colors['red'])
    ax.set_xscale('log')
    ax.set_ylim(0.0, 0.25*np.pi)
    ax.set_yticks([0.0, 0.25*np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/4$'])
    ax.set_xlabel('Frequency [Hz]')
    fig.savefig('filter-fratebode')
    
        
if __name__ == "__main__":
    plot_whitenoise()
    plot_stimulus()
    plot_stimulus_psd()
    plot_rate_psd()
    plot_ratetransfer()


    
