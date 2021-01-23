import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def lifac_spikes(time, stimulus, taum=0.01, tref=0.003, noisedv=0.01, vreset=0.0, vthresh=1.0,
                 taua=0.1, alpha=0.05, noiseda=0.0, rng=np.random):
    """ Leaky integrate-and-fire neuron with adaptation current.

    $$\\tau_m \\frac{dV}{dt} = - V + RI - A + D\\xi$$

    All default time constants are in seconds.

    Parameters
    ----------
    time: 1D array
        Time vector. Sets the integration time step.
    stimulus: 1D array
        For each time point in `time` the stimulus value (in units of the membrane voltage!
        This is $$R I$$).
    taum: float
        Membrane time constant, same unit as `time`.
    tref: float
        Absolute refractory period (spike width), same unit as `time`.
    noisedv: float
        Noise strength D of additive white noise for membrane equation.
    vreset: float
        Reset value for membrane voltage.
    vthresh: float
        Threshold value for membrane voltage. When the voltage crosses this threshold
        from below a spike is generated and the voltage is reset to `vreset`.
    taua: float
        Adaptation time constant, same unit as `time`.
    alpha: float
        Adaptation strength. At each spike the adaptation variable is incremented by
        `alpha` divided by `taua`.
    noiseda: float
        Noise strength D of additive white noise for adaptation dynamics.
    rng: random number generator with an randn() function
        Random number generator for computing the additive noise.
              
    Returns
    -------
    spikes: 1D array
        List of spike times.
    """
    # time step:
    dt = time[1] - time[0]
    # noise terms properly scaled:
    noisev = rng.randn(len(stimulus))*noisedv/np.sqrt(dt)
    noisea = rng.randn(len(stimulus))*noiseda/np.sqrt(dt)
    # initializiation for forgetting initial conditions:
    tn = time[0]
    V = rng.rand()*(vthresh-vreset) + vreset
    A = 0.0
    for k in range(min(1000, len(noisev))):
        if time[k] < tn:
            continue
        V += (-V - A + stimulus[0] + noisev[k])*dt/taum
        A += (-A + noisea[k])*dt/taua
        if V > vthresh:
            V = vreset
            A += alpha/taua
            tn = time[k] + tref
    # integration:
    tn = time[0]
    spikes = []
    for k in range(len(stimulus)):
        # no integration during refractory period:
        if time[k] < tn:
            continue
        # membrane equation:
        V += (-V - A + stimulus[k] + noisev[k])*dt/taum
        # adaptation dynamics:
        A += (-A + noisea[k])*dt/taua
        # threshold condition:
        if V > vthresh:
            V = vreset               # voltage reset
            A += alpha/taua          # adaptation increment
            tn = time[k] + tref      # refractory period
            spikes.append(time[k])   # store spike time
    return np.asarray(spikes)


def plot_sparse(ax1, ax2, ax3, ax4):
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
        spks = lifac_spikes(time, 0.95+1.2*ge*stimulus, taua=0.1, taum=0.01, alpha=0.0,
                            noisedv=0.01)
        spikes.append(1000.0*spks)
        spks = lifac_spikes(time, 0.95+gei*stimulus, taua=0.1, taum=0.01, alpha=0.0,
                            noisedv=0.01)
        spikesi.append(1000.0*spks)
        spks = lifac_spikes(time, 12.0+1.2*16.6*ge*stimulus, taua=0.1, taum=0.01, alpha=0.7,
                            noisedv=0.2)
        spikesa.append(1000.0*spks)
        spks = lifac_spikes(time, 0.0+16.6*gei*stimulus, taua=0.1, taum=0.01, alpha=0.7,
                            noisedv=0.2)
        spikesai.append(1000.0*spks)
    # plotting in milliseconds:
    time *= 1000.0
    ton *= 1000.0
    toff *= 1000.0
    # plain:
    ax1.set_title('no adaptation, no inhibition')
    ax1.add_patch(mpl.patches.Rectangle((0.0, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax1.add_patch(mpl.patches.Rectangle((toff, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax1.eventplot(spikes, colors=['k'], lw=0.5)
    ax1.set_xlim(time[0], time[-1])
    ax1.set_ylabel('neuron')
    # inhibition:
    ax2.set_title('no adaptation, lateral inhibition')
    ax2.add_patch(mpl.patches.Rectangle((0.0, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax2.add_patch(mpl.patches.Rectangle((toff, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax2.eventplot(spikesi, colors=['k'], lw=0.5)
    ax2.set_xlim(time[0], time[-1])
    # adaptation:
    ax3.set_title('adaptation, no inhibition')
    ax3.add_patch(mpl.patches.Rectangle((0.0, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax3.add_patch(mpl.patches.Rectangle((toff, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax3.eventplot(spikesa, colors=['k'], lw=0.5)
    ax3.set_xlim(time[0], time[-1])
    ax3.set_xlabel('time [ms]')
    ax3.set_ylabel('neuron')
    # both:
    ax4.set_title('adaptation, lateral inhibition')
    ax4.add_patch(mpl.patches.Rectangle((0.0, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax4.add_patch(mpl.patches.Rectangle((toff, -0.5), ton, n+0.5, ec='none', lw=0, fc='#EEEEEE'))
    ax4.eventplot(spikesai, colors=['k'], lw=0.5)
    ax4.set_xlim(time[0], time[-1])
    ax4.set_xlabel('time [ms]')


def sparse_demo():
    """ Demo of generation of sparse codes.
    """
    plt.rcParams['axes.xmargin'] = 0.0
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    plot_sparse(*axs.ravel())
    plt.show()

        
if __name__ == "__main__":
    sparse_demo()
    
