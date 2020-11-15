import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def lifac(time, stimulus, taum=0.01, tref=0.003, noisedv=0.01, vreset=0.0, vthresh=1.0,
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
    VV: 1D array
        Membrane voltage.
    AA: 1D array
        Adaptation variable.
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
    VV = np.zeros(len(stimulus))
    AA = np.zeros(len(stimulus))
    tn = time[0]
    spikes = []
    for k in range(len(stimulus)):
        VV[k] = V
        AA[k] = A
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
    return np.asarray(spikes), VV, AA


def spike_frequency(time, spikes, fill=0.0):
    """ Instantaneous firing rate (1/ISI) averaged over trials.

    Parameters
    ----------
    time: 1D array
        Time vector on which to evaluate the firing rate.
    spikes: list of 1D-arrays of float
        For each trial the spike times.
    fill: float or 'extend'
        How to fill in values for the firing rate before the first and
        after the last spike.  If 'extend' then fill up with the border
        firing rates, otherwise use the provided value.
        Trials with less than two spikes get rates with the fill value.
    
    Returns
    -------
    frate: 1D array
        The trial-averaged firing rate.
    """
    zv = 0.0 if fill == 'extend' else fill
    rates = np.zeros((len(time), len(spikes)))
    for k in range(len(spikes)):
        isis = np.diff(spikes[k])
        if len(spikes[k]) > 2:
            fv = (1.0/isis[0], 1.0/isis[-1]) if fill == 'extend' else (fill, fill)
            fr = interp1d(spikes[k][:-1], 1.0/isis, kind='previous',
                          bounds_error=False, fill_value=fv)
            rate = fr(time)
        else:
            rate = np.zeros(len(time)) + zv
        rates[:,k] = rate
    frate = np.mean(rates, 1)
    return frate


def step_stimulus(tmin, tmax, dur, s0, s1):
    """ Step stimulus with time vector.
    
    Returns
    -------
    time: 1D array
        Time vector.
    stimulus: 1D array
        Step stimulus.
    """
    dt = 0.0001               # integration time step in seconds
    time = np.arange(tmin, tmax+dt, dt)
    stimulus = np.zeros(len(time)) + s0
    stimulus[(time > 0.0) & (time < dur)] = s1
    return time, stimulus


def baseline_activity(s, tmax, model, **kwargs):
    """ Simulate neuron model with a fixed stimulus.

    Parameters
    ----------
    s: float
        Stimulus value for the model.
    tmax: float
        Maximum integration time.
    model: function
        The model.
    kwargs: dict
        Parameter for the model.

    Returns
    -------
    spikes: 1D arrays of floats
        The spike times.
    """
    dt = 0.0001                         # integration time step in seconds
    time = np.arange(0.0, tmax, dt)
    stimulus = np.zeros(len(time)) + s
    spikes, _, _ = model(time, stimulus, **kwargs)
    return spikes[spikes > 1.0] - 1.0   # steady-state only

    
def plot_lifac_trial(ax, time, stimulus):
    """ Plot stimulus, membrane voltage, adaptation current and spikes for a single simulated trial of the LIFAC.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to plot.
    time: 1D array of float
        Time array.
    stimulus: 1D array of float
        Stimulus.
    """
    tfac = 1000.0             # plots in milliseconds
    spikes, V, A = lifac(time, stimulus)
    ax.axhline(0.0, color='k', linestyle=':', label=r'$V_r$')
    ax.axhline(1.0, color='k', linestyle='--', label=r'$\theta$')
    ax.plot(tfac*time, stimulus, label='$RI(t)$')
    ax.plot(tfac*time, V, label='$V(t)$')
    ax.plot(tfac*time, A, label='$A(t)$')
    ax.eventplot([tfac*spikes], colors=['k'], lineoffsets=1.6)
    ax.set_xlim(tfac*time[0], tfac*time[-1])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Membrane voltage')
    ax.legend()


def plot_raster(ax, time, spikes):
    """ Plot a spike raster.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to plot.
    time: 1D array of float
        Time array.
    spikes: list of 1D array of float
        For each trial an array of spike times.
    """
    tfac = 1000.0             # plots in milliseconds
    spks = [tfac*spikes[k] for k in range(len(spikes))]
    ax.eventplot(spks, colors=['k'], lineoffsets=np.arange(1, len(spks)+1))
    ax.set_xlim(tfac*time[0], tfac*time[-1])
    ax.set_ylim(0, len(spks)+1)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Trials')

    
def plot_spike_frequency(ax, time, spikes):
    """ Plot instantaneous firing rate computed from spikes.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to plot.
    time: 1D array of float
        Time array.
    spikes: list of 1D array of float
        For each trial an array of spike times.
    """
    tfac = 1000.0             # plots in milliseconds
    ratetime = np.arange(time[0], time[-1], 0.001)
    frate = spike_frequency(ratetime, spikes, 'extend')
    ax.plot(tfac*ratetime, frate)
    ax.set_xlim(tfac*time[0], tfac*time[-1])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Firing rate [Hz]')


def plot_isih(ax, spikes, labels=[]):
    """ Interspike-interval histograms for each trial.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to plot.
    spikes: list of 1D array of float
        For each trial an array of spike times.
    labels: list of string
        For each trial a label describing the condition.
    """
    tfac = 1000.0             # plots in milliseconds
    bw = 0.0005               # bin width in seconds
    for spks, l in zip(spikes, labels):
        isis = np.diff(spks)
        bins = np.arange((np.min(isis)//bw)*bw, (np.max(isis)//bw+1)*bw, bw)
        ax.hist(tfac*isis, tfac*bins, density=True, label=l)
    ax.set_xlabel('ISI [ms]')
    ax.set_ylabel('pdf [kHz]')
    ax.legend()

 
def plot_serial_correlation(ax, spikes, labels=[], max_lag=5):
    """ Serial correlations of interspike intervals for each trial.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to plot.
    spikes: list of 1D array of float
        For each trial an array of spike times.
    labels: list of string
        For each trial a label describing the condition.
    max_lag: int
        Maximum lag for which a serial correlation is computed.
    """
    ax.axhline(0.0, color='k', linestyle=':')
    for spks, l in zip(spikes, labels):
        isis = np.diff(spks)
        lags = np.arange(0, max_lag+1)
        scorr = [1.0] + [np.corrcoef(isis[k:], isis[:-k])[1,0] for k in lags[1:]]
        ax.plot(lags, scorr, '-o', label=l)
    ax.set_ylim(-0.5, 1)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Serial correlation')
    ax.legend()

    
def plot_lifac_fIcurves(ax):
    """ F-I curves of the LIFAC.

    Parameters
    ----------
    ax: matplotlib axes
        Axes where to plot.
    """
    n = 20                    # number of trials
    dt = 0.0001               # integration time step in seconds
    inputs = np.arange(0, 10.1, 0.5)
    fon = np.zeros(len(inputs))
    fss = np.zeros(len(inputs))
    fa = np.zeros(len(inputs))
    # onset and steady-state f-I curve:
    time = np.arange(0.0, 0.3, dt)
    stimulus = np.zeros(len(time))
    ratetime = np.arange(time[0], time[-1], 0.001)
    for i, stim in enumerate(inputs):
        stimulus[time > 0.0] = stim
        spikes = [lifac(time, stimulus)[0] for k in range(20)]
        frate = spike_frequency(ratetime, spikes, 'extend')
        fon[i] = np.max(frate)
        fss[i] = np.mean(frate[(ratetime>0.2) & (ratetime<0.25)])
    # adapted f-I curve:
    prestim = 4.0
    time = np.arange(-0.5, 0.3, dt)
    stimulus = np.zeros(len(time)) + prestim
    ratetime = np.arange(time[0], time[-1], 0.001)
    for i, stim in enumerate(inputs):
        stimulus[time > 0.0] = stim
        spikes = [lifac(time, stimulus)[0] for k in range(20)]
        frate = spike_frequency(ratetime, spikes)
        baserate = np.mean(frate[(ratetime>-0.1) & (ratetime<0.0)])
        arate = frate[(ratetime>0.0) & (ratetime<0.1)]
        inx = np.argmax(np.abs(arate-baserate))
        fa[i] = arate[inx]
    ax.plot(inputs, fss, 'r', label=r'$f_{\infty}(I)$')
    ax.plot(inputs, fon, 'g', label=r'$f_{0}(I)$')
    ax.plot(inputs, fa, 'b', label=r'$f_{0}(I-A)$')
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Firing rate [Hz]')
    ax.legend()


def lifac_demo():
    """ Demo of the LIFAC and the functions in this module.
    """
    fig, axs = plt.subplots(3, 2, constrained_layout=True)
    # step response:
    time, stimulus = step_stimulus(-0.2, 0.8, 0.2, 1.2, 4.0)
    spikes = [lifac(time, stimulus)[0] for k in range(20)]
    stimulus = stimulus[time<0.6]
    time = time[time<0.6]
    plot_lifac_trial(axs[0,0], time, stimulus)
    plot_raster(axs[1,0], time, spikes)
    plot_spike_frequency(axs[2,0], time, spikes)
    # f-I curves:
    plot_lifac_fIcurves(axs[2,1])
    # baseline statistics:
    inputs = [2.0, 4.0, 6.0]
    spikes = [baseline_activity(s, 200.0, lifac, noiseda=0.03) for s in inputs]
    labels = ['RI=%g' % s for s in inputs]
    plot_isih(axs[0,1], spikes, labels)
    plot_serial_correlation(axs[1,1], spikes, labels)
    plt.show()

        
if __name__ == "__main__":
    lifac_demo()
    
