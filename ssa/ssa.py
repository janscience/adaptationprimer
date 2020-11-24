import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def upper_boltzmann(x, ymax=1.0, x0=0.0, slope=1.0):
    """ Upper half of Boltzmann as a sigmoidal function.

    Parameters
    ----------
    x: float or array
        X-value(s) for which to compute the sigmoidal function.
    ymax: float
        Maximum y-value at saturation.
    x0: float
        Position (foot point) of the sigmoidal.
    slope: float
        Slope factor of the sigmoidal.

    Returns
    -------
    y: float or array
        The sigmoidal function for each x value.
    """
    y = ymax*(2.0/(1.0+np.exp(-slope*(x-x0))) - 1.0)
    if np.isscalar(y):
        if y < 0.0:
            y = 0.0
    else:
        y[y<0] = 0.0
    return y


def adaptation(time, stimulus, taua=0.1, alpha=1.0, slope=1.0, I0=0.0, fmax=200.0):
    """ Subtractive spike-frequency of an adaptating neuron with sigmoidal onset f-I curve.

    Parameters
    ----------
    time: 1D array
        Time vector. Sets the integration time step.
    stimulus: 1D array
        For each time point in `time` the stimulus value.
    taua: float
        Adaptation time constant, same unit as `time`.
    alpha: float
        Adaptation strength.
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
    dt = time[1] - time[0]
    # integrate to steady-state of first stimulus value:
    a = 0.0
    for k in range(int(5*taua//dt)):
        f = upper_boltzmann(stimulus[0] - a, fmax, I0, slope)
        a += (alpha*f - a)*dt/taua
    # integrate:
    rate = np.zeros(len(stimulus))
    adapt = np.zeros(len(stimulus))
    for k in range(len(stimulus)):
        adapt[k] = a
        rate[k] = f
        f = upper_boltzmann(stimulus[k] - a, fmax, I0, slope)
        a += (alpha*f - a)*dt/taua
    return rate, adapt


def plot_pulseadaptation(axs, axr):
    """ Adaptation to pulse stimulus.
    """
    tfac = 1000.0
    n = 5              # number of pulses
    T = 0.1            # period of the pulses in seconds
    t0 = 0.03          # start of the pulse within the period in seconds
    t1 = 0.07          # end of the pulse within the period in seconds
    dt = 0.0005        # integration time step in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = np.zeros(len(time))
    stimulus[(time%T>t0) & (time%T<t1)] = 2.0
    rate, adapt = adaptation(time, stimulus, alpha=0.2, taua=1.0)
    # plot:
    axs.set_title('Standard')
    axs.plot(time, stimulus, label='stimulus')
    axs.plot(time, adapt, label='threshold')
    axs.set_ylabel('Stimulus')
    axs.legend(loc='upper left')
    axr.plot(time, rate, label='adapting')
    axr.set_xlabel('Time [s]')
    axr.set_ylabel('Spike frequency [Hz]')
    

def plot_sawtoothstimulus(ax):
    """ Periodic sawtooth stimulus.
    """
    tfac = 1000.0
    n = 5              # number of pulses
    T = 0.1            # period of the pulses in seconds
    dt = 0.0005        # integration time step in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = 1.0 - (time%T)/T
    # plot:
    ax.plot(time, stimulus, label='stimulus')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    

def plot_cosinestimulus(ax):
    """ Periodic cosine stimulus.
    """
    tfac = 1000.0
    n = 5              # number of pulses
    T = 0.1            # period of the pulses in seconds
    dt = 0.0005        # integration time step in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = 0.5*(1.0 - np.cos(2.0*np.pi*time/T))
    stimulus = stimulus**2
    # plot:
    ax.plot(time, stimulus, label='stimulus')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    

def plot_deviant(ax):
    """ Rare cosine pulses.
    """
    tfac = 1000.0
    n = 20             # number of pulses
    m = 5              # deviant on every m-th pulse
    T = 0.1            # period of the pulses in seconds
    dt = 0.0005        # integration time step in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = 0.5*(1.0 - np.cos(2.0*np.pi*time/T))
    stimulus = stimulus**2
    deviant = np.array(stimulus)
    deviant[time%(m*T) < (m-1)*T] = 0.0
    # plot:
    ax.plot(time, stimulus, label='stimulus')
    ax.plot(time, deviant, label='deviant')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Deviant')
    

def plot_deviantadaptation(axs, axr):
    """ Adaptation to rare cosine pulses.
    """
    tfac = 1000.0
    n = 20             # number of pulses
    m = 5              # deviant on every m-th pulse
    T = 0.1            # period of the pulses in seconds
    t0 = 0.03          # start of the pulse within the period in seconds
    t1 = 0.07          # end of the pulse within the period in seconds
    dt = 0.0005        # integration time step in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = np.zeros(len(time))
    stimulus[(time%T>t0) & (time%T<t1)] = 2.0
    deviant = np.array(stimulus)
    deviant[time%(m*T) < (m-1)*T] = 0.0
    rate, adapt = adaptation(time, deviant, alpha=0.2, taua=1.0)
    # plot:
    axs.set_title('Deviant')
    axs.plot(time, deviant, label='stimulus')
    axs.plot(time, adapt, label='threshold')
    axs.set_ylabel('Stimulus')
    axs.legend(loc='upper left')
    axr.plot(time, rate, label='adapting')
    axr.set_xlabel('Time [s]')
    axr.set_ylabel('Spike frequency [Hz]')
    

def plot_ssa(axs, axr):
    """ SSA.
    """
    tfac = 1000.0
    n = 20             # number of pulses
    m = 5              # deviant on every m-th pulse
    T = 0.1            # period of the pulses in seconds
    t0 = 0.03          # start of the pulse within the period in seconds
    t1 = 0.07          # end of the pulse within the period in seconds
    dt = 0.0005        # integration time step in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = np.zeros(len(time))
    stimulus[(time%T>t0) & (time%T<t1)] = 2.0
    deviant = np.array(stimulus)
    deviant[time%(m*T) < (m-1)*T] = 0.0
    # responses:
    rates, adapts = adaptation(time, stimulus, alpha=0.2, taua=1.0)
    rated, adaptd = adaptation(time, deviant, alpha=0.2, taua=1.0)
    rate = rates + rated
    # plot:
    axs.set_title('SSA')
    axs.plot(time, stimulus, label='standard')
    axs.plot(time, deviant, label='deviant')
    axs.set_ylabel('Stimulus')
    axs.legend(loc='upper left')
    axr.plot(time, rate, label='adapting')
    axr.set_xlabel('Time [s]')
    axr.set_ylabel('Spike frequency [Hz]')
    

def ssa_demo():
    """ Demo of stimulus-specific adaptation.
    """
    plt.rcParams['axes.xmargin'] = 0.0
    fig, axs = plt.subplots(2, 4, constrained_layout=True)
    plot_pulseadaptation(axs[0,0], axs[1,0])
    plot_cosinestimulus(axs[0,1])
    plot_deviant(axs[1,1])
    plot_deviantadaptation(axs[0,2], axs[1,2])
    plot_ssa(axs[0,3], axs[1,3])
    plt.show()

        
if __name__ == "__main__":
    ssa_demo()
    
