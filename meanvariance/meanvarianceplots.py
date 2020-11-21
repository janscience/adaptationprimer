import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import matplotlib.ticker as ticker
import meanvariance as mv
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors


def plot_meanstimulus():
    dt = 0.001                    # integration time step in seconds
    tmax = 4.0                    # stimulus duration in seconds
    cutoff = 20.0                 # cutoff frequency of stimulus in Hertz
    T = 1.0                       # duration of segements with constant mean in seconds
    means = [0.0, 3.0, 6.0, 1.5]  # mean stimulus values for each segment
    rng = np.random.RandomState(583)
    stimulus = 0.5*mv.whitenoise(0.0, cutoff, dt, tmax, rng)
    time = np.arange(len(stimulus))*dt
    mean = np.zeros(len(stimulus))
    for k, m in enumerate(means):
        mean[(time>k*T) & (time<=(k+1)*T)] += m
    stimulus += mean
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(time, stimulus, color=colors['green'])
    ax.plot(time, mean, '--', color=colors['lightgreen'])
    ax.set_ylim(-1.5, 7.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    fig.savefig('meanvariance-meanstimulus')
    # response:
    rate0, _ = mv.adaptation(time, stimulus, alpha=0.0, taua=0.5)
    rate, _ = mv.adaptation(time, stimulus, alpha=0.2, taua=0.5)
    # plot:
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(time, rate0, color=colors['cyan'], label='non adapting')
    ax.plot(time, rate, color=colors['blue'], label='adapting')
    ax.set_ylim(0.0, 200.0)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Spike frequency [Hz]')
    ax.legend(loc='upper left')
    fig.savefig('meanvariance-meanresponse')
    
        
if __name__ == "__main__":
    plot_meanstimulus()


    
