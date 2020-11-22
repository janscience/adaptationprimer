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


def divisive_adaptation(time, stimulus, taua=0.1, slope=1.0, I0=0.0, fmax=200.0):
    """ Divisive spike-frequency of an adaptating neuron with sigmoidal onset f-I curve.

    Parameters
    ----------
    time: 1D array
        Time vector. Sets the integration time step.
    stimulus: 1D array
        For each time point in `time` the stimulus value.
    taua: float
        Adaptation time constant, same unit as `time`.
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
    #alpha = 0.2
    dt = time[1] - time[0]
    # integrate to steady-state of first stimulus value:
    a = 1.0
    for k in range(int(5*taua//dt)):
        f = upper_boltzmann(stimulus[0]/a, fmax, I0, slope)
        #a += (alpha*f - a)*dt/taua
        a += (stimulus[0] - a)*dt/taua
    # integrate:
    rate = np.zeros(len(stimulus))
    adapt = np.zeros(len(stimulus))
    for k in range(len(stimulus)):
        f = upper_boltzmann(stimulus[k]/a, fmax, I0, slope)
        #a += (alpha*f - a)*dt/taua
        a += (stimulus[k] - a)*dt/taua
        adapt[k] = a
        rate[k] = f
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


def amplitude_modulation(signal, dt, fcutoff):
    """ Extract amplitude modulation.

    The signal is first threshoded at zero and then low-pass filtered.

    Parameters
    ----------
    signal: 1D array
        The signal, a carrier with an amplitude modulation.
    dt: float
        Time step of the signal.
    fcutoff: float
        Cutoff frequency for low-pass filter.

    Returns
    -------
    am: 1D array
        Amplitude modulation of the signal.
    """
    signal = np.array(signal)
    signal[signal<0.0] = 0.0
    sos = sig.butter(2, fcutoff, 'lp', fs=1.0/dt, output='sos')
    am = 2.0*sig.sosfilt(sos, signal)
    return am


def plot_meanstimulus(axs, axr):
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoff = 20.0                 # cutoff frequency of stimulus in Hertz
    T = 1.0                       # duration of segements with constant mean in seconds
    means = [0.0, 3.0, 6.0, 1.5]  # mean stimulus values for each segment
    stimulus = 0.5*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    mean = np.zeros(len(stimulus))
    for k, m in enumerate(means):
        mean[(time>k*T) & (time<=(k+1)*T)] += m
    stimulus += mean
    # response of non adapting neuron:
    rate0, adapt0 = adaptation(time, stimulus, alpha=0.0, taua=0.5)
    # response of adapting neuron:
    rate, adapt = adaptation(time, stimulus, alpha=0.2, taua=0.5)
    # plot:
    axs.set_title('Mean steps')
    axs.plot(time, stimulus, label='stimulus')
    axs.plot(time, adapt, label='threshold')
    axs.set_ylabel('Stimulus')
    axs.legend(loc='upper left')
    axr.plot(time, rate0, label='non adapting')
    axr.plot(time, rate, label='adapting')
    axr.set_xlabel('Time [s]')
    axr.set_ylabel('Spike frequency [Hz]')
    axr.legend(loc='upper left')
    

def plot_meansine(axs, axr):
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoff = 40.0                 # cutoff frequency of stimulus in Hertz
    stimulus = 0.5*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    mean = 3.0*(1.0-np.cos(2.0*np.pi*time/time[-1]))
    stimulus += mean
    # response of non adapting neuron:
    rate0, adapt0 = adaptation(time, stimulus, alpha=0.0, taua=0.5)
    # response of adapting neuron:
    rate, adapt = adaptation(time, stimulus, alpha=0.2, taua=0.5)
    # plot:
    axs.set_title('Slow cosine')
    axs.plot(time, stimulus, label='stimulus')
    axs.plot(time, adapt, label='threshold')
    axs.set_ylabel('Stimulus')
    axs.legend(loc='upper left')
    axr.plot(time, rate0, label='non adapting')
    axr.plot(time, rate, label='adapting')
    axr.set_xlabel('Time [s]')
    axr.set_ylabel('Spike frequency [Hz]')
    axr.legend(loc='upper left')
    

def plot_variancestimulus(axs, axr):
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoff = 60.0                 # cutoff frequency of stimulus in Hertz
    T = 1.0                       # duration of segements with constant mean in seconds
    stdevs = [0.5, 1.5, 3.0, 0.5] # standard deviations for each segment
    stimulus = 0.5*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    std = np.zeros(len(stimulus))
    for k, s in enumerate(stdevs):
        std[(time>k*T) & (time<=(k+1)*T)] += s
    stimulus *= std
    stimulus += 2.0
    # response of non adapting neuron:
    rate0, adapt0 = adaptation(time, stimulus, alpha=0.0, taua=0.5)
    # response of adapting neuron:
    rate, adapt = adaptation(time, stimulus, alpha=0.2, taua=0.5)
    # plot:
    axs.set_title('Variance steps')
    axs.plot(time, stimulus, label='stimulus')
    axs.plot(time, adapt, label='threshold')
    axs.set_ylabel('Stimulus')
    axs.legend(loc='upper left')
    axr.plot(time, rate0, label='non adapting')
    axr.plot(time, rate, label='adapting')
    axr.set_xlabel('Time [s]')
    axr.set_ylabel('Spike frequency [Hz]')
    axr.legend(loc='upper left')
    

def plot_amplitudemodulation(axs):
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoffl = 40.0                # lower cutoff frequency of stimulus in Hertz
    cutoffu = 80.0                # upper cutoff frequency of stimulus in Hertz
    T = 1.0                       # duration of segements with constant mean in seconds
    stdevs = [0.5, 1.5, 3.0, 0.5] # standard deviations for each segment
    #stimulus = 0.5*whitenoise(cutoffl, cutoffu, dt, tmax)
    #time = np.arange(len(stimulus))*dt
    time = np.arange(0.0, tmax, dt)
    stimulus = 0.5*np.sin(2.0*np.pi*cutoffl*time)
    std = np.zeros(len(stimulus))
    for k, s in enumerate(stdevs):
        std[(time>k*T) & (time<=(k+1)*T)] += s
    stimulus *= std
    stimulus += 2.0
    # compute amplitude modulation:
    stimulus -= np.mean(stimulus)
    am = amplitude_modulation(stimulus, dt, 2.0)
    # power spectra:
    freqs, pstim = sig.welch(stimulus, fs=1.0/dt, nperseg=2**11)
    stim = np.array(stimulus)
    stim[stim<0.0] = 0.0
    freqs, pthresh = sig.welch(stim, fs=1.0/dt, nperseg=2**11)
    freqs, pam = sig.welch(am, fs=1.0/dt, nperseg=2**11)
    # plot:
    ax = axs[0]
    ax.plot(time, stimulus, label='stimulus')
    ax.plot(time, am, label='AM')
    ax.set_ylabel('Stimulus')
    ax.legend(loc='upper left')
    ax = axs[1]
    ax.plot(freqs, 10.0*np.log10(pstim/np.max(pstim)))
    ax.set_xlim(0, 100.0)
    ax.set_ylim(-30, 0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power stimulus [dB]')
    ax = axs[2]
    ax.plot(freqs, 10.0*np.log10(pthresh/np.max(pthresh)))
    ax.plot(freqs, 10.0*np.log10(0.5*pam/np.max(pthresh)))
    ax.set_xlim(0, 100.0)
    ax.set_ylim(-30, 0)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power thresh. stimulus [dB]')
    

def plot_divisivevariance(axs, axr):
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoff = 60.0                 # cutoff frequency of stimulus in Hertz
    T = 1.0                       # duration of segements with constant mean in seconds
    stdevs = [0.5, 1.5, 3.0, 0.5] # standard deviations for each segment
    stimulus = 0.5*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    std = np.zeros(len(stimulus))
    for k, s in enumerate(stdevs):
        std[(time>k*T) & (time<=(k+1)*T)] += s
    stimulus *= std
    stimulus[stimulus<0.0] = 0.0
    # response of non adapting neuron:
    rate0, adapt0 = adaptation(time, stimulus, alpha=0.2, taua=0.3, slope=0.5)
    # response of adapting neuron:
    rate, adapt = divisive_adaptation(time, stimulus, taua=0.3, slope=0.1)
    # plot:
    axs.set_title('Divisive variance')
    axs.plot(time, adapt, label='threshold')
    axs.set_ylabel('Adaptation')
    axr.plot(time, rate0, label='non adapting')
    axr.plot(time, rate, label='adapting')
    axr.set_xlabel('Time [s]')
    axr.set_ylabel('Spike frequency [Hz]')
    axr.legend(loc='upper left')
    

def meanvariance_demo():
    """ Demo of adaptation to the mean and variance of a stimulus.
    """
    plt.rcParams['axes.xmargin'] = 0.0
    fig, axs = plt.subplots(2, 3, constrained_layout=True)
    plot_meanstimulus(axs[0,0], axs[1,0])
    plot_meansine(axs[0,1], axs[1,1])
    #plot_variancestimulus(axs[0,2], axs[1,2])
    plot_divisivevariance(axs[0,2], axs[1,2])
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    plot_amplitudemodulation(axs)
    plt.show()

        
if __name__ == "__main__":
    meanvariance_demo()
    
