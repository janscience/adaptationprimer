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
    output: 1D array
        The resulting time course of the spike frequncy.
    adapt: 1D array
        The time course of the adaptation level.
    """
    f0 = lambda I: fmax*(2.0/(1.0+np.exp(-slope*(I-I0))) - 1.0) if I > I0 else 0.0
    dt = time[1] - time[0]
    output = np.zeros(len(stimulus))
    adapt = np.zeros(len(stimulus))
    a = 0.0
    f = 0.0
    for k in range(int(3*taua//dt)):
        a += (alpha*f - a)*dt/taua
        f = f0(stimulus[0] - a)
    for k in range(len(stimulus)):
        adapt[k] = a
        output[k] = f
        a += (alpha*f - a)*dt/taua
        f = f0(stimulus[k] - a)
    return output, adapt


def plot_step(axf, axa):
    """ Plot spike frequency and adaptation level in response to step.
    """
    tfac = 1000.0             # plots in milliseconds
    dt = 0.0001               # integration time step in seconds
    time = np.arange(-0.05, 0.3+dt, dt)
    stimulus = np.zeros(len(time))
    stimulus[(time > 0.0) & (time < 0.1)] = 1.0
    frate, adapt = adaptation_sigmoid(time, stimulus, alpha=0.05)
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
    
