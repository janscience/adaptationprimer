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
    """ Adabtation to pulse stimulus.
    """
    n = 5
    T = 0.1
    time = np.arange(0.0, n*T, 0.0005)
    stimulus = np.zeros(len(time)) + 0.0
    for k in range(n):
        stimulus[(time>k*T+0.025) & (time<k*T+0.075)] = 2.0
    rate, adapt = adaptation(time, stimulus, alpha=0.1, taua=0.3)
    # plot:
    axs.plot(time, stimulus, label='stimulus')
    axs.plot(time, adapt, label='threshold')
    axs.set_ylabel('Stimulus')
    axs.legend(loc='upper left')
    axr.plot(time, rate, label='adapting')
    axr.set_xlabel('Time [s]')
    axr.set_ylabel('Spike frequency [Hz]')
    

def ssa_demo():
    """ Demo of stimulus-specific adaptation.
    """
    plt.rcParams['axes.xmargin'] = 0.0
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    plot_pulseadaptation(axs[0,0], axs[1,0])
    plt.show()

        
if __name__ == "__main__":
    ssa_demo()
    
