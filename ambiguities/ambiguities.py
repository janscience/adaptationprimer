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
    # zero and nyquist frequency must be real:
    if inx0 == 0:
        whitef[0] = 0
        inx0 = 1
    if inx1 >= nn//2:
        whitef[nn//2] = 1
        inx1 = nn//2-1
    phases = 2*np.pi*rng.rand(inx1 - inx0 + 1)
    whitef[inx0:inx1+1] = np.cos(phases) + 1j*np.sin(phases)
    # inverse FFT:
    noise = np.real(np.fft.irfft(whitef))
    # scaling factor to ensure standard deviation of one:
    sigma = nn / np.sqrt(2*float(inx1 - inx0))
    return noise[:n]*sigma


def plot_ambiguities(axr, axs, axd, axa):
    # amplitude modulation stimulus for left ear:
    dt = 0.005
    tmax = 1.0
    am_left = 5.0*(1.0 + 0.3*whitenoise(0.0, 20.0, dt, tmax))
    time = np.arange(len(am_left))*dt
    am_left[time<0.1] = 0.0
    # loop through three attenuations of right ear:
    decibels = [8.0, 4.0, 0.0]
    for k, dbs in enumerate(decibels):
        ampl = 10**(-0.1*dbs)     # attenuation factor from decibels
        am_right = ampl*am_left   # attenuate stimulus for right ear
        # convert to decibels (receptors are logarithmic):
        db_left = 10.0*np.log10(am_left)
        db_right = 10.0*np.log10(am_right)
        # adapting receptor responses:
        rate_left, _ = adaptation(time, db_left, taua=0.5, alpha=0.4)
        rate_right, _ = adaptation(time, db_right, taua=0.5, alpha=0.4)
        # neuronal noise for left and right receptor:
        rate_left += 10.0*whitenoise(0.0, 60.0, dt, tmax)
        rate_right += 10.0*whitenoise(0.0, 60.0, dt, tmax)
        # sum and difference responses:
        rate_sum = rate_left + rate_right
        rate_diff = rate_left - rate_right
        rate_diff_adapt, _ = adaptation(time, rate_diff, taua=10.0, alpha=50.0, slope=0.01)
        # plot:
        axr.plot(time, rate_right)
        axs.plot(time, rate_sum)
        axd.plot(time, rate_diff)
        axa.plot(time, rate_diff_adapt, label='%g dB' % dbs)
    axr.set_title('Receptor')
    axr.set_ylabel('Spike frequency [Hz]')
    axs.set_title('Sum')
    axd.set_title('Difference')
    axd.set_ylabel('Spike frequency [Hz]')
    axd.set_xlabel('Time [s]')
    axa.set_title('Difference adapted')
    axa.set_xlabel('Time [s]')
    axa.legend()


def ambiguities_demo():
    """ Demo of stimulus-specific adaptation.
    """
    plt.rcParams['axes.xmargin'] = 0.0
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    plot_ambiguities(*axs.ravel())
    plt.show()

        
if __name__ == "__main__":
    ambiguities_demo()
    
