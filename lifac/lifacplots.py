import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lifac as la
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth


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
    #ax.plot(tfac*time, stimulus, label='$RI(t)$')
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
    fig, ax = plt.subplots(figsize=(figwidth, 0.6*figwidth))
    ax.eventplot(spks, colors=['k'], lineoffsets=np.arange(1, len(spks)+1), lw=0.5)
    ax.set_xlim(tfac*time[0], tfac*time[-1])
    ax.set_ylim(0, len(spks)+1)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Trials')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5.0))
    fig.savefig('lifac-raster')


if __name__ == "__main__":
    time, stimulus = la.step_stimulus(-0.2, 0.8, 0.2, 1.2, 4.0)
    plot_stimulus(time, stimulus)
    plot_trial(time, stimulus)
    spikes = [la.lifac(time, stimulus)[0] for k in range(20)]
    plot_raster(time, spikes)


