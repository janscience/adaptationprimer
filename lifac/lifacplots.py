import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lifac as la
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors


def plot_stimulus(time, stimulus):
    tfac = 1000.0
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(tfac*time, stimulus, label=r'$RI(t)$')
    ax.set_ylim(0, 4.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Stimulus [mV]')
    ax.legend()
    fig.savefig('lifac-stepstimulus')

    
def plot_trial(time, stimulus):
    tfac = 1000.0             # plots in milliseconds
    spikes, V, A = la.lifac(time, stimulus)
    fig, ax = plt.subplots(figsize=(figwidth, 0.6*figwidth))
    ax.axhline(0.0, color='k', linestyle=':', label=r'$V_r$')
    ax.axhline(1.0, color='k', linestyle='--', label=r'$\theta$')
    ax.plot(tfac*time, V, label='$V(t)$')
    ax.plot(tfac*time, A, label='$A(t)$')
    ax.eventplot([tfac*spikes], colors=['k'], lineoffsets=1.6)
    ax.set_xlim(tfac*time[0], tfac*time[-1])
    ax.set_ylim(-1.5, 3.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Membrane voltage [mV]')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.legend(ncol=2)
    fig.savefig('lifac-trial')


def plot_raster(time, spikes):
    tfac = 1000.0             # plots in milliseconds
    spks = [tfac*spikes[k] for k in range(len(spikes))]
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.eventplot(spks, colors=['k'], lineoffsets=np.arange(1, len(spks)+1), lw=0.5)
    ax.set_xlim(tfac*time[0], tfac*time[-1])
    ax.set_ylim(0, len(spks)+1)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Trials')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5.0))
    fig.savefig('lifac-raster')

    
def plot_spike_frequency(time, spikes):
    tfac = 1000.0             # plots in milliseconds
    ratetime = np.arange(time[0], time[-1], 0.001)
    frate = la.spike_frequency(ratetime, spikes, 'extend')
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.plot(tfac*ratetime, frate)
    ax.set_xlim(tfac*time[0], tfac*time[-1])
    ax.set_ylim(0, 150)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50.0))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Firing rate [Hz]')
    fig.savefig('lifac-rate')

    
def plot_lifac_fIcurves():
    n = 20                    # number of trials
    dt = 0.0001               # integration time step in seconds
    inputs = np.arange(0, 10.1, 0.2)
    fon = np.zeros(len(inputs))
    fss = np.zeros(len(inputs))
    fa = np.zeros(len(inputs))
    # onset and steady-state f-I curve:
    time = np.arange(0.0, 0.3, dt)
    stimulus = np.zeros(len(time))
    ratetime = np.arange(time[0], time[-1], 0.001)
    for i, stim in enumerate(inputs):
        stimulus[time > 0.0] = stim
        spikes = [la.lifac(time, stimulus)[0] for k in range(20)]
        frate = la.spike_frequency(ratetime, spikes, 'extend')
        fon[i] = np.max(frate)
        fss[i] = np.mean(frate[(ratetime>0.2) & (ratetime<0.25)])
    # adapted f-I curve:
    prestim = 4.0
    time = np.arange(-0.5, 0.3, dt)
    stimulus = np.zeros(len(time)) + prestim
    ratetime = np.arange(time[0], time[-1], 0.001)
    for i, stim in enumerate(inputs):
        stimulus[time > 0.0] = stim
        spikes = [la.lifac(time, stimulus)[0] for k in range(20)]
        frate = la.spike_frequency(ratetime, spikes)
        baserate = np.mean(frate[(ratetime>-0.1) & (ratetime<0.0)])
        arate = frate[(ratetime>0.0) & (ratetime<0.1)]
        inx = np.argmax(np.abs(arate-baserate))
        fa[i] = arate[inx]
    fig, ax = plt.subplots(figsize=(figwidth, 0.6*figwidth))
    ax.plot(inputs, fss, colors['red'], label=r'$f_{\infty}(I)$')
    ax.plot(inputs, fon, colors['green'], label=r'$f_{0}(I)$')
    ax.plot(inputs, fa, colors['blue'], label=r'$f_{a}(I)$')
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Firing rate [Hz]')
    ax.set_ylim(0, 250)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50.0))
    ax.legend()
    fig.savefig('lifac-ficurves')


def baseline_activity(s, tmax):
    dt = 0.0001               # integration time step in seconds
    time = np.arange(0.0, tmax, dt)
    stimulus = np.zeros(len(time)) + s
    spikes, _, _ = la.lifac(time, stimulus, noiseda=0.03)
    return spikes[spikes > 1.0] - 1.0   # steady-state only


def plot_isih(spikes, labels=[]):
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    tfac = 1000.0             # plots in milliseconds
    bw = 0.0004               # bin width in seconds
    for spks, l in zip(spikes, labels):
        isis = np.diff(spks)
        bins = np.arange((np.min(isis)//bw)*bw, (np.max(isis)//bw+1)*bw, bw)
        ax.hist(tfac*isis, tfac*bins, density=True, label=l)
    ax.set_xlim(0, 70)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel('ISI [ms]')
    ax.set_ylabel('pdf [kHz]')
    ax.legend()
    fig.savefig('lifac-isih')

 
def plot_serial_correlation(spikes, labels=[], max_lag=5):
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.axhline(0.0, color='k', linestyle=':')
    for spks, l in zip(spikes, labels):
        isis = np.diff(spks)
        lags = np.arange(0, max_lag+1)
        scorr = [1.0] + [np.corrcoef(isis[k:], isis[:-k])[1,0] for k in lags[1:]]
        ax.plot(lags, scorr, '-o', label=l, clip_on=False)
    ax.set_ylim(-0.5, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel('Lag')
    ax.set_ylabel('Serial correlation')
    ax.legend()
    fig.savefig('lifac-isicorr')


if __name__ == "__main__":
    # step responses:
    time, stimulus = la.step_stimulus(-0.2, 0.8, 0.2, 1.2, 4.0)
    plot_stimulus(time, stimulus)
    plot_trial(time, stimulus)
    spikes = [la.lifac(time, stimulus)[0] for k in range(20)]
    plot_raster(time, spikes)
    plot_spike_frequency(time, spikes)
    plot_lifac_fIcurves()
    # baseline statistics:
    inputs = [2.0, 4.0, 8.0]
    spikes = [baseline_activity(s, 200.0) for s in inputs]
    labels = ['RI=%g' % s for s in inputs]
    plot_isih(spikes, labels)
    plot_serial_correlation(spikes, labels)


