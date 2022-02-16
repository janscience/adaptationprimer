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


def transfer(stimulus, response, dt, nfft=2**10, maxf=None):
    """ Transfer function between stimulus and response.

    Parameters
    ----------
    stimulus: 1D array
        Stimulus time series.
    response: 1D array
        Response time series corresponding to `stimulus`.
    dt: float
        Time step in seconds.
    nfft: int
        Number of samples used for Fourier transformation. Ideally a power of two.
    maxf: float or None
        If not `None`, truncate returned arrays at this frequency (in Hertz).
        This usually is the cutoff frequency of the stimulus used.

    Returns
    -------
    freqs: 1D array
        Frequencies in Hertz.
    gain: 1D array
        Gain of the transfer function for each frequency in `freqs`.
    phase: 1D array
        Phase of the transfer function in radians for each frequency in `freqs`.
    """
    freqs, csd = sig.csd(stimulus, response, fs=1.0/dt, nperseg=nfft)
    freqs, psd = sig.welch(stimulus, fs=1.0/dt, nperseg=nfft)
    transfer = csd/psd
    if maxf:
        transfer = transfer[freqs<maxf]
        freqs = freqs[freqs<maxf]
    gain = np.abs(transfer)
    phase = np.angle(transfer)
    return freqs, gain, phase


def plot_whitenoise(ax):
    dt = 0.001              # integration time step in seconds
    tmax = 1.0              # stimulus duration in seconds
    cutoff = 50.0           # highest frequency in stimulus in Hertz
    stimulus = whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    ax.plot(time, stimulus)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('White noise')
    

def plot_stimulus1(ax):
    dt = 0.001              # integration time step in seconds
    tmax = 1.0              # stimulus duration in seconds
    cutoff = 50.0           # highest frequency in stimulus in Hertz
    mean = 5.0              # stimulus mean
    stdev = 2.5             # stimulus standard deviation
    stimulus = mean + stdev*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    ax.plot(time, stimulus)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    

def plot_stimulus2(ax):
    dt = 0.001              # integration time step in seconds
    tmax = 1.0              # stimulus duration in seconds
    cutoff = 10.0           # highest frequency in stimulus in Hertz
    mean = 5.0              # stimulus mean
    stdev = 2.5             # stimulus standard deviation
    stimulus = mean + stdev*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    ax.plot(time, stimulus)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    

def plot_transfer(axf, axfl, axfp, axa, axal, axap):
    """ Plot transfer fucntion for spike frequency and adaptation level.
    """
    dt = 0.0001               # integration time step in seconds
    tmax = 100.0              # stimulus duration
    cutoff = 1010.0           # highest frequency in stimulus
    nfft = 2**12              # number of samples for Fourier trafo
    stimulus = 6.0 + 2.0*whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    rate, adapt = adaptation(time, stimulus, alpha=0.05, taua=0.02)
    frate = isi_lowpass(time, rate)
    freqs, rgain, rphase = transfer(stimulus, rate, dt, nfft, cutoff)
    freqs, fgain, fphase = transfer(stimulus, frate, dt, nfft, cutoff)
    freqs, again, aphase = transfer(stimulus, adapt, dt, nfft, cutoff)
    axf.plot(freqs, rgain, label=r'$f(t)$')
    axf.plot(freqs, fgain, label=r'$\langle f(t) \rangle $')
    axf.set_ylabel('Rate gain')
    axf.legend()
    axfl.plot(freqs, rgain)
    axfl.plot(freqs, fgain)
    axfl.set_xscale('log')
    axfl.set_yscale('log')
    axfl.set_ylim(1e0, 1e2)
    axfp.plot(freqs, rphase)
    axfp.plot(freqs, fphase)
    axfp.set_xscale('log')
    axfp.set_xlim(1, 100)
    axfp.set_ylim(0, 1)
    axfp.set_ylabel('Rate phase')
    axa.plot(freqs, again)
    axa.set_xlabel('Frequency [Hz]')
    axa.set_ylabel('Adaptation gain')
    axal.plot(freqs, again)
    axal.set_xscale('log')
    axal.set_yscale('log')
    axal.set_ylim(1e-2, 1e0)
    axal.set_xlabel('Frequency [Hz]')
    axap.plot(freqs, aphase)
    axap.set_xscale('log')
    axap.set_xlim(1, 100)
    axap.set_xlabel('Frequency [Hz]')
    axap.set_ylabel('Adaptation phase')


def filter_demo():
    """ Demo of the filter properties of spike-frequency adaptation.
    """
    plt.rcParams['axes.xmargin'] = 0.0
    fig, axs = plt.subplots(3, 3, constrained_layout=True)
    plot_whitenoise(axs[0,0])
    plot_stimulus1(axs[0,1])
    plot_stimulus2(axs[0,2])
    plot_transfer(axs[1,0], axs[1,1], axs[1,2], axs[2,0], axs[2,1], axs[2,2])
    plt.show()

        
if __name__ == "__main__":
    filter_demo()
    
