import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sparse as sp
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors


def plot_spatialprofile():
    n = 20                              # number of neurons
    ids = np.arange(0, n, 1)            # neuron ids
    activity = np.exp(-0.5*((ids-0.5*n)/(0.1*n))**2)
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.plot(ids, activity, '-o', color=colors['red'], clip_on=False)
    ax.set_xticks(np.arange(0, n+1, 5))
    ax.set_xlabel('Neuron Id')
    ax.set_ylabel('Activity')
    fig.savefig('sparse-spatialprofile')


def plot_spatialinhibition():
    n = 20                              # number of neurons
    ids = np.arange(0, n, 1)            # neuron ids
    activity = 2.4*np.exp(-0.5*((ids-0.5*n)/(0.1*n))**2) - 1.0
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.axhline(0.0, linestyle='--', color=colors['gray'])
    ax.plot(ids, activity, '-o', color=colors['blue'], clip_on=False)
    ax.set_xticks(np.arange(0, n+1, 5))
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel('Neuron Id')
    ax.set_ylabel('Activity')
    fig.savefig('sparse-spatialinhibition')


def plot_sparse():
    n = 20                              # number of neurons
    ton = 0.15                          # onset time of stimulus
    toff = 2*ton                        # offset time of stimulus
    dt = 0.0005                         # integration time step
    time = np.arange(0.0, toff+ton, dt)
    stimulus = np.zeros(len(time))      # rectangular stimulus
    stimulus[(time>ton) & (time<toff)] = 1.0
    spikes = []
    spikesa = []
    spikesi = []
    spikesai = []
    # loop over all the neurons:
    for k in range(n):
        # Gaussian excitation profile modelling the stimulus input:
        ge = np.exp(-0.5*((k-0.5*n)/(0.1*n))**2)
        # Lateral inhibition simply reduces the excitation:
        gei = 2.4*ge - 1.0
        spks = sp.lifac_spikes(time, 0.95+1.2*ge*stimulus, taua=0.1, taum=0.01, alpha=0.0,
                               noisedv=0.01)
        spikes.append(1000.0*spks)
        spks = sp.lifac_spikes(time, 0.95+gei*stimulus, taua=0.1, taum=0.01, alpha=0.0,
                               noisedv=0.01)
        spikesi.append(1000.0*spks)
        spks = sp.lifac_spikes(time, 12.0+1.2*16.6*ge*stimulus, taua=0.1, taum=0.01, alpha=0.7,
                               noisedv=0.2)
        spikesa.append(1000.0*spks)
        spks = sp.lifac_spikes(time, 0.0+16.6*gei*stimulus, taua=0.1, taum=0.01, alpha=0.7,
                               noisedv=0.2)
        spikesai.append(1000.0*spks)
    # plotting in milliseconds:
    time *= 1000.0
    ton *= 1000.0
    toff *= 1000.0
    # plain:
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.set_title('no adaptation, no inhibition')
    ax.add_patch(mpl.patches.Rectangle((0.0, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax.add_patch(mpl.patches.Rectangle((toff, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax.eventplot(spikes, colors=['k'], lw=0.5)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('neuron')
    fig.savefig('sparse-plain')
    # inhibition:
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.set_title('no adaptation, lateral inhibition')
    ax.add_patch(mpl.patches.Rectangle((0.0, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax.add_patch(mpl.patches.Rectangle((toff, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax.eventplot(spikesi, colors=['k'], lw=0.5)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('neuron')
    fig.savefig('sparse-inhibition')
    # adaptation:
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.set_title('adaptation, no inhibition')
    ax.add_patch(mpl.patches.Rectangle((0.0, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax.add_patch(mpl.patches.Rectangle((toff, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax.eventplot(spikesa, colors=['k'], lw=0.5)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('neuron')
    fig.savefig('sparse-adaptation')
    # both:
    fig, ax = plt.subplots(figsize=(figwidth, 0.5*figwidth))
    ax.set_title('adaptation, lateral inhibition')
    ax.add_patch(mpl.patches.Rectangle((0.0, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax.add_patch(mpl.patches.Rectangle((toff, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax.eventplot(spikesai, colors=['k'], lw=0.5)
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('neuron')
    fig.savefig('sparse-both')

        
if __name__ == "__main__":
    plot_spatialprofile()
    plot_spatialinhibition()
    plot_sparse()
    
