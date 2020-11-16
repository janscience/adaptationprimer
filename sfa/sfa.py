import numpy as np
import matplotlib.pyplot as plt


def adaptation_sigmoid(time, stimulus, taua=0.1, alpha=1.0, slope=1.0, I0=0.0, fmax=200.0):
    """ Spike-frequency of an adaptating neuron with sigmoidal onset f-I curve.

    Parameters
    ----------
    time: 1D array
        Time vector. Sets the integration time step.
    stimulus: 1D array
        For each time point in `time` the stimulus value.
    taua: float
        Adaptation time constant, same unit as `time`.
    alpha: float
        Adaptation strength. At each spike the adaptation variable is incremented by
        `alpha` divided by `taua`.
    slope: float
        Slope of the sigmoidal f-I curve.
    I0: float
        Position of the sigmoidal f-I curve.
    fmax: float
        Maximum spike frequency of the onset f-I curve.  

    Returns
    -------
    rate: 1D array
        The resulting time course of the spike frequency.
    adapt: 1D array
        The time course of the adaptation level.
    """
    # sigmoidal onset f-I curve:
    f0 = lambda I: fmax*(2.0/(1.0+np.exp(-slope*(I-I0))) - 1.0) if I > I0 else 0.0
    dt = time[1] - time[0]
    # integrate to steady-state of first stimulus value:
    a = 0.0
    for k in range(int(3*taua//dt)):
        f = f0(stimulus[0] - a)
        a += (alpha*f - a)*dt/taua
    # integrate:
    rate = np.zeros(len(stimulus))
    adapt = np.zeros(len(stimulus))
    for k in range(len(stimulus)):
        adapt[k] = a
        rate[k] = f
        f = f0(stimulus[k] - a)
        a += (alpha*f - a)*dt/taua
    return rate, adapt


def isi_lowpass(time, rate, time_centered=False):
    """ Limit-cycle firing low-pass filter.

    For each point in `time` integrate `rate` symmetrically to the left
    and the right until the integral equals one. The inverse of the
    required integration range is then the returned spike frequency.
    
    Parameters
    ----------
    time: 1D array
        Time vector. Sets the integration time step.
    rate: 1D array
        Time course of the spike frequency from the adaptation model.
    time_centered: bool
        If `True` then integrate symmetrically to the left and right.
        If `False` then integrate to both sides half a period. 

    Returns
    -------
    frate: 1D array
        Spike frequency low pass filtered over one ISI length.
    """
    dt = time[1] - time[0]                           # integration time step
    # extend spike frequency on both ends to avoid edge effects:
    lidx = int(np.ceil(1.0/rate[0]/dt))+1 if rate[0] > 0.0 else len(rate)
    ridx = int(np.ceil(1.0/rate[-1]/dt))+1 if rate[-1] > 0.0 else len(rate)
    rr = np.hstack(([rate[0]]*lidx, rate, [rate[-1]]*ridx))
    tt = np.arange(len(rr))*dt + time[0] - lidx*dt   # new time array  
    # integrate the firing rate to a phase:
    phi = np.cumsum(rr)*dt
    t0 = np.interp(phi-0.5, phi, tt)
    t1 = np.interp(phi+0.5, phi, tt)
    frate = 1.0/(t1 - t0)
    if time_centered:
        tc = (t0 + t1)/2.0
        return np.interp(time, tc, frate)
    else:
        return frate[lidx:-ridx]


def plot_step(axf, axa):
    """ Plot spike frequency and adaptation level in response to step.
    """
    tfac = 1000.0             # plots in milliseconds
    dt = 0.0001               # integration time step in seconds
    time = np.arange(-0.2, 0.5+dt, dt)
    stimulus = np.zeros(len(time)) + 1.0
    stimulus[(time > 0.0) & (time < 0.1)] = 3.0
    rate, adapt = adaptation_sigmoid(time, stimulus, alpha=0.05)
    frate = isi_lowpass(time, rate)
    axf.plot(tfac*time, rate)
    axf.plot(tfac*time, frate)
    axf.set_xlabel('Time [ms]')
    axf.set_ylabel('Spike frequency [Hz]')
    axa.plot(tfac*time, adapt)
    axa.set_xlabel('Time [ms]')
    axa.set_ylabel('Adaptation')


def sfa_demo():
    """ Demo of the spike-frequency adaptation model and the functions in this module.
    """
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    plot_step(axs[0,0], axs[1,0])
    plt.show()

        
if __name__ == "__main__":
    sfa_demo()
    
