import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def lifac(time, stimulus, taum=0.01, tref=0.003, noised=0.01,
          vreset=0.0, vthresh=1.0, taua=0.1, alpha=0.05, rng=np.random):
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
    noised: float
        Noise strength D.
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
    # noise term properly scaled:
    noise = rng.randn(len(stimulus))*noised/np.sqrt(dt)
    # initializiation for forgetting initial conditions:
    tn = time[0]
    V = rng.rand()
    A = 0.0
    for k in range(min(1000, len(noise))):
        if time[k] < tn:
            continue
        V += (-V - A + stimulus[0] + noise[k])*dt/taum
        A += -A*dt/taua
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
        V += (-V - A + stimulus[k] + noise[k])*dt/taum
        # adaptation dynamics:
        A += -A*dt/taua
        # threshold condition:
        if V > vthresh:
            V = vreset               # voltage reset
            A += alpha/taua          # adaptation increment
            tn = time[k] + tref      # refractory period
            spikes.append(time[k])   # store spike time
    return np.asarray(spikes), VV, AA


def firing_rate(time, spikes, fill=0.0):
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


def plot_lifac_trial(ax):
    tfac = 1000.0             # plots in milliseconds
    dt = 0.0001               # integration time step in seconds
    time = np.arange(-0.2, 0.6, dt)
    # step stimulus:
    stimulus = np.zeros(len(time)) + 1.2
    stimulus[(time > 0.0) & (time < 0.2)] = 4.0
    # single trial:
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


def plot_lifac_raster(ax):
    tfac = 1000.0             # plots in milliseconds
    dt = 0.0001               # integration time step in seconds
    time = np.arange(-0.2, 0.5, dt)
    # step stimulus:
    stimulus = np.zeros(len(time)) + 1.2
    stimulus[(time > 0.0) & (time < 0.2)] = 4.0
    # raster:
    n = 20
    spikes = []
    for k in range(n):
        s, _, _ = lifac(time, stimulus)
        spikes.append(tfac*s)
    ax.eventplot(spikes, colors=['k'], lineoffsets=np.arange(1, n+1))
    ax.set_xlim(tfac*time[0], tfac*time[-1])
    ax.set_ylim(0, n+1)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Trials')

    
def plot_lifac_serial_correlation(ax):
    dt = 0.0001               # integration time step in seconds
    time = np.arange(0.0, 200.0, dt)    # long integration for good statistics!
    stimulus = np.zeros(len(time)) + 2.0
    spikes, _, _ = lifac(time, stimulus)
    isis = np.diff(spikes)
    lags = np.arange(0, 6)
    scorr = [1.0] + [np.corrcoef(isis[k:], isis[:-k])[1,0] for k in lags[1:]]
    ax.axhline(0.0, color='k', linestyle=':')
    ax.plot(lags, scorr, '-o', color='r')
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Serial correlation')

    
def plot_lifac_rate(ax):
    n = 20                    # number of trials
    tfac = 1000.0             # plots in milliseconds
    dt = 0.0001               # integration time step in seconds
    time = np.arange(-0.2, 0.6, dt)
    # step stimulus:
    stimulus = np.zeros(len(time)) + 1.2
    stimulus[(time > 0.0) & (time < 0.2)] = 4.0
    ratetime = np.arange(time[0], time[-1], 0.001)
    spikes = []
    for k in range(n):
        s, _, _ = lifac(time, stimulus)
        spikes.append(s)
    frate = firing_rate(ratetime, spikes)
    ax.plot(ratetime, frate)

    
def plot_lifac_prerate(ax):
    n = 20                    # number of trials
    tfac = 1000.0             # plots in milliseconds
    dt = 0.0001               # integration time step in seconds
    time = np.arange(-0.2, 0.6, dt)
    # step stimulus:
    stimulus = np.zeros(len(time))
    stimulus[(time > 0.0) & (time < 0.2)] = 4.0
    stimulus[(time > 0.2) & (time < 0.4)] = 8.0
    ratetime = np.arange(time[0], time[-1], 0.001)
    spikes = []
    for k in range(n):
        s, _, _ = lifac(time, stimulus)
        spikes.append(s)
    frate = firing_rate(ratetime, spikes)
    #ax.eventplot(spikes, colors=['k'], lineoffsets=np.arange(1, n+1))
    #ax.plot(time, stimulus)
    ax.plot(ratetime, frate)

    
def plot_lifac_fIcurves(ax):
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
        spikes = []
        for k in range(n):
            s, _, _ = lifac(time, stimulus)
            spikes.append(s)
        frate = firing_rate(ratetime, spikes)
        fon[i] = np.max(frate)
        fss[i] = np.mean(frate[(ratetime>0.2)])
    # adapted f-I curve:
    prestim = 4.0
    time = np.arange(-0.5, 0.3, dt)
    stimulus = np.zeros(len(time)) + prestim
    ratetime = np.arange(time[0], time[-1], 0.001)
    for i, stim in enumerate(inputs):
        stimulus[time > 0.0] = stim
        spikes = []
        for k in range(n):
            s, _, _ = lifac(time, stimulus)
            spikes.append(s)
        frate = firing_rate(ratetime, spikes)
        baserate = np.mean(frate[(ratetime>-0.1) & (ratetime<0.0)])
        inx = np.argmax(np.abs(frate[(ratetime>0.002) & (ratetime<0.1)]-baserate))
        fa[i] = frate[(ratetime>0.002) & (ratetime<0.1)][inx]
        #plt.plot(ratetime, frate)
        #plt.show()
    ax.plot(inputs, fss, 'r', label=r'$f_{\infty}(I)$')
    ax.plot(inputs, fon, 'g', label=r'$f_{0}(I)$')
    ax.plot(inputs, fa, 'b', label=r'$f_{0}(I-A)$')
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Firing rate [Hz]')

        
if __name__ == "__main__":
    fig, axs = plt.subplots(3, 2)
    #plot_lifac_trial(axs[0,0])
    #plot_lifac_raster(axs[1,0])
    #plot_lifac_rate(axs[2,0])
    #plot_lifac_serial_correlation(axs[0,1])
    #plot_lifac_prerate(axs[1,1])
    plot_lifac_fIcurves(axs[2,1])
    plt.show()
    
