import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ssa
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors


def plot_pulseadaptation():
    tfac = 1000.0
    n = 5
    T = 0.1
    dt = 0.0005
    time = np.arange(0.0, n*T, dt)
    stimulus = np.zeros(len(time))
    stimulus[(time%T>t0) & (time%T<t1)] = 2.0
    rate, adapt = ssa.adaptation(time, stimulus, alpha=0.2, taua=1.0)
    # plot:
    fig, axs = plt.subplots(2, 1, figsize=(figwidth, 0.6*figwidth))
    ax = axs[0]
    ax.plot(tfac*time, stimulus, color=colors['green'], label='stimulus')
    ax.plot(tfac*time, adapt, color=colors['red'], label='threshold')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.set_ylabel('Stimulus')
    ax.legend(loc='upper left')
    ax = axs[1]
    ax.plot(tfac*time, rate, color=colors['blue'])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Spike frequency [Hz]')
    fig.savefig('ssa-pulseadaptation')
    

if __name__ == "__main__":
    plot_pulseadaptation()
