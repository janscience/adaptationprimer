import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import matplotlib.ticker as ticker
import meanvariance as mv
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors


def plot_meanstimulus():
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoff = 20.0                 # cutoff frequency of stimulus in Hertz
    T = 1.0                       # duration of segements with constant mean in seconds
    means = [0.0, 3.0, 6.0, 1.5]  # mean stimulus values for each segment
    rng = np.random.RandomState(583)
    stimulus = 0.5*mv.whitenoise(0.0, cutoff, dt, tmax, rng)
    time = np.arange(len(stimulus))*dt
    mean = np.zeros(len(stimulus))
    for k, m in enumerate(means):
        mean[(time>k*T) & (time<=(k+1)*T)] += m
    stimulus += mean
    # plot stimulus:
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(time, stimulus, color=colors['green'])
    ax.plot(time, mean, '--', color=colors['lightgreen'])
    ax.set_ylim(-1.5, 7.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    fig.savefig('meanvariance-meanstimulus')
    # response:
    rate0, adapt0 = mv.adaptation(time, stimulus, alpha=0.0, taua=0.5)
    rate, adapt = mv.adaptation(time, stimulus, alpha=0.2, taua=0.5)
    # plot rate:
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(time, rate0, color=colors['cyan'], label='non adapting')
    ax.plot(time, rate, color=colors['blue'], label='adapting')
    ax.set_ylim(0.0, 200.0)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Spike frequency [Hz]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.legend(loc='upper left')
    fig.savefig('meanvariance-meanresponse')
    # plot adaptation:
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(time, adapt, color=colors['red'], label='adapting')
    ax.set_ylim(0.0, 8.0)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Adaptation')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    fig.savefig('meanvariance-meanadapt')
    # plot stimulus and threshold:
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(time, stimulus, color=colors['green'], label='stimulus')
    ax.fill_between(time, adapt, -1.5, fc=colors['gray'], alpha=0.75)
    ax.plot(time, adapt, color=colors['red'], label='threshold')
    ax.set_ylim(-1.5, 7.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.legend(loc='upper left')
    fig.savefig('meanvariance-meanthreshold')
    

def plot_meansine():
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoff = 40.0                 # cutoff frequency of stimulus in Hertz
    rng = np.random.RandomState(583)
    stimulus = 0.5*mv.whitenoise(0.0, cutoff, dt, tmax, rng)
    time = np.arange(len(stimulus))*dt
    mean = 3.0*(1.0-np.cos(2.0*np.pi*time/time[-1]))
    stimulus += mean
    # response:
    rate0, adapt0 = mv.adaptation(time, stimulus, alpha=0.0, taua=0.3)
    rate, adapt = mv.adaptation(time, stimulus, alpha=0.2, taua=0.3)
    # plot stimulus and threshold:
    fig, axs = plt.subplots(2, 1, figsize=(figwidth, 0.6*figwidth))
    ax = axs[0]
    ax.plot(time, stimulus, color=colors['green'], label='stimulus')
    ax.fill_between(time, adapt, -1.5, fc=colors['gray'], alpha=0.75)
    ax.plot(time, adapt, color=colors['red'], label='threshold')
    ax.set_ylim(-1.5, 7.5)
    ax.set_ylabel('Stimulus')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.legend(loc='upper left')
    # plot rate:
    ax = axs[1]
    ax.plot(time, rate0, color=colors['cyan'], label='non adapting')
    ax.plot(time, rate, color=colors['blue'], label='adapting')
    ax.set_ylim(0.0, 200.0)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Spike frequency [Hz]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.legend(loc='upper left')
    fig.savefig('meanvariance-meansine')
    

def plot_variancestimulus():
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoff = 60.0                 # cutoff frequency of stimulus in Hertz
    T = 1.0                       # duration of segements with constant mean in seconds
    stdevs = [0.5, 1.5, 3.0, 0.5] # standard deviations for each segment
    rng = np.random.RandomState(583)
    stimulus = 0.5*mv.whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    std = np.zeros(len(stimulus))
    for k, s in enumerate(stdevs):
        std[(time>k*T) & (time<=(k+1)*T)] += s
    stimulus *= std
    stimulus += 2.0
    # response:
    rate0, adapt0 = mv.adaptation(time, stimulus, alpha=0.0, taua=0.3)
    rate, adapt = mv.adaptation(time, stimulus, alpha=0.2, taua=0.3)
    # plot stimulus and threshold:
    fig, axs = plt.subplots(2, 1, figsize=(figwidth, 0.6*figwidth))
    ax = axs[0]
    ax.plot(time, stimulus, color=colors['green'], label='stimulus')
    ax.fill_between(time, adapt, -1.5, fc=colors['gray'], alpha=0.75)
    ax.plot(time, adapt, color=colors['red'], label='threshold')
    ax.set_ylim(-1.5, 7.5)
    ax.set_ylabel('Stimulus')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.legend(loc='upper left')
    # plot rate:
    ax = axs[1]
    ax.plot(time, rate0, color=colors['cyan'], label='non adapting')
    ax.plot(time, rate, color=colors['blue'], label='adapting')
    ax.set_ylim(0.0, 200.0)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Spike frequency [Hz]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.legend(loc='upper left')
    fig.savefig('meanvariance-variancethreshold')
    

def plot_amplitudemodulation():
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoffl = 40.0                # lower cutoff frequency of stimulus in Hertz
    cutoffu = 80.0                # upper cutoff frequency of stimulus in Hertz
    T = 1.0                       # duration of segements with constant mean in seconds
    stdevs = [0.5, 1.5, 3.0, 0.5] # standard deviations for each segment
    #stimulus = 0.5*mv.whitenoise(cutoffl, cutoffu, dt, tmax)
    #time = np.arange(len(stimulus))*dt
    time = np.arange(0.0, tmax, dt)
    stimulus = 0.5*np.sin(2.0*np.pi*cutoffl*time)
    std = np.zeros(len(stimulus))
    for k, s in enumerate(stdevs):
        std[(time>k*T) & (time<=(k+1)*T)] += s
    stimulus *= std
    stimulus += 2.0
    # compute amplitude modulation:
    stimulus -= np.mean(stimulus)
    am = mv.amplitude_modulation(stimulus, dt, 2.0)
    # power spectra:
    nfft = 2**11
    freqs, pstim = sig.welch(stimulus, fs=1.0/dt, nperseg=nfft)
    stim = np.array(stimulus)
    stim[stim<0.0] = 0.0
    freqs, pthresh = sig.welch(stim, fs=1.0/dt, nperseg=nfft)
    freqs, pam = sig.welch(am, fs=1.0/dt, nperseg=nfft)
    # plot:
    fig, axs = plt.subplots(3, 1, figsize=(figwidth, 0.8*figwidth))
    ax = axs[0]
    ax.plot(time, stimulus, label='stimulus')
    ax.plot(time, am, label='AM')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stimulus')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.2))
    ax = axs[1]
    ax.plot(freqs, 10.0*np.log10(pstim/np.max(pstim)), label='stimulus')
    ax.set_xlim(0, 100.0)
    ax.set_ylim(-30, 0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.2))
    ax = axs[2]
    ax.plot(freqs, 10.0*np.log10(pthresh/np.max(pthresh)), label='thresholded stimulus')
    ax.plot(freqs, 10.0*np.log10(pam/np.max(pthresh)), label='AM')
    ax.set_xlim(0, 100.0)
    ax.set_ylim(-30, 0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.4))
    fig.savefig('meanvariance-amplitudemodulation')
    
        
if __name__ == "__main__":
    plot_meanstimulus()
    plot_meansine()
    plot_variancestimulus()
    plot_amplitudemodulation()


    
