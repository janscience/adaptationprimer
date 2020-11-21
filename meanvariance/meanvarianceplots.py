import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import matplotlib.ticker as ticker
import meanvariance as mv
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors


def plot_meanstimulus():
    dt = 0.001              # integration time step in seconds
    tmax = 4.0              # stimulus duration in seconds
    cutoff = 50.0           # highest frequency in stimulus in Hertz
    rng = np.random.RandomState(583)
    stimulus = 0.5*mv.whitenoise(0.0, cutoff, dt, tmax)
    time = np.arange(len(stimulus))*dt
    mean = np.zeros(len(stimulus))
    mean[(time>1.0) & (time<=2.0)] += 3.0
    mean[(time>2.0) & (time<=3.0)] += 6.0
    mean[(time>3.0) & (time<=4.0)] += 1.5
    stimulus += mean
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(time, stimulus, color=colors['green'])
    ax.plot(time, mean, '--', color=colors['lightgreen'])
    ax.set_ylim(-1.5, 7.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Stimulus')
    fig.savefig('meanvariance-meanstimulus')
    
        
if __name__ == "__main__":
    plot_meanstimulus()


    
