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


def whitenoise(cflow, cfup, dt, duration, rng=np.random):
    """Band-limited white noise.

    Generates white noise with a flat power spectrum between `cflow` and
    `cfup` Hertz, zero mean and unit standard deviation.  Note, that in
    particular for short segments of the generated noise the mean and
    standard deviation of the returned noise can deviate from zero and
    one.

    Parameters
    ----------
    cflow: float
        Lower cutoff frequency in Hertz.
    cfup: float
        Upper cutoff frequency in Hertz.
    dt: float
        Time step of the resulting array in seconds.
    duration: float
        Total duration of the resulting array in seconds.

    Returns
    -------
    noise: 1-D array
        White noise.
    """
    # number of elements needed for the noise stimulus:
    n = int(np.ceil((duration+0.5*dt)/dt))
    # next power of two:
    nn = int(2**(np.ceil(np.log2(n))))
    # indices of frequencies with `cflow` and `cfup`:
    inx0 = int(np.round(dt*nn*cflow))
    inx1 = int(np.round(dt*nn*cfup))
    if inx0 < 0:
        inx0 = 0
    if inx1 >= nn/2:
        inx1 = nn/2
    # draw random numbers in Fourier domain:
    whitef = np.zeros((nn//2+1), dtype=complex)
    if inx0 == 0:
        whitef[0] = rng.randn()
        inx0 = 1
    if inx1 >= nn//2:
        whitef[nn//2] = rng.randn()
        inx1 = nn//2-1
    m = inx1 - inx0 + 1
    whitef[inx0:inx1+1] = rng.randn(m) + 1j*rng.randn(m)
    # scaling factor to ensure standard deviation of one:
    sigma = 0.5 / np.sqrt(float(inx1 - inx0))
    # inverse FFT:
    noise = np.real(np.fft.irfft(whitef))[:n]*sigma*nn
    return noise


def plot_meanstimulus(ax):
    dt = 0.001              # integration time step in seconds
    tmax = 4.0              # stimulus duration in seconds
    cutoff = 50.0           # cutoff frequency of stimulus in Hertz
    stimulus = 0.5*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    mean = np.zeros(len(stimulus))
    mean[(time>1.0) & (time<=2.0)] += 3.0
    mean[(time>2.0) & (time<=3.0)] += 6.0
    mean[(time>3.0) & (time<=4.0)] += 1.5
    stimulus += mean
    ax.plot(time, stimulus)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    

def meanvariance_demo():
    """ Demo of adaptation to the mean and variance of a stimulus.
    """
    plt.rcParams['axes.xmargin'] = 0.0
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    plot_meanstimulus(axs[0,0])
    plt.show()

        
if __name__ == "__main__":
    meanvariance_demo()
    
