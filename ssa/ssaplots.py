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
    t0 = 0.03
    t1 = 0.07
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
    

def plot_sawtoothstimulus():
    tfac = 1000.0
    n = 5              # number of pulses
    T = 0.1            # period of the pulses in seconds
    dt = 0.0005        # integration time step in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = 1.0 - (time%T)/T
    # plot:
    fig, ax = plt.subplots(figsize=(figwidth, 0.3*figwidth))
    ax.plot(tfac*time, stimulus, color=colors['green'], clip_on=False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Stimulus')
    fig.savefig('ssa-sawtoothstimulus')
    

def plot_cosinestimulus():
    tfac = 1000.0
    n = 5              # number of pulses
    T = 0.1            # period of the pulses in seconds
    dt = 0.0005        # integration time step in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = 0.5*(1.0 - np.cos(2.0*np.pi*time/T))
    stimulus = stimulus**2
    # plot:
    fig, ax = plt.subplots(figsize=(figwidth, 0.3*figwidth))
    ax.plot(tfac*time, stimulus, color=colors['green'], clip_on=False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Stimulus')
    fig.savefig('ssa-cosinestimulus')
    

def plot_deviant():
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
    fig, ax = plt.subplots(figsize=(figwidth, 0.3*figwidth))
    ax.plot(tfac*time, stimulus, color=colors['green'], clip_on=False)
    ax.plot(tfac*time, deviant, color=colors['orange'], clip_on=False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Stimulus')
    fig.savefig('ssa-deviant')
    

def plot_deviantadaptation():
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
    rate, adapt = ssa.adaptation(time, deviant, alpha=0.2, taua=1.0)
    # plot:
    fig, axs = plt.subplots(2, 1, figsize=(figwidth, 0.6*figwidth))
    ax = axs[0]
    ax.plot(time, deviant, color=colors['orange'], label='deviant', clip_on=False)
    ax.plot(time, adapt, color=colors['red'], label='threshold')
    ax.set_ylabel('Stimulus')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.legend(loc='upper left')
    ax = axs[1]
    ax.plot(time, rate, color=colors['blue'], label='adapting')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Spike frequency [Hz]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    fig.savefig('ssa-deviantadaptation')
    

def plot_ssa():
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
    rates, adapts = ssa.adaptation(time, stimulus, alpha=0.2, taua=1.0)
    rated, adaptd = ssa.adaptation(time, deviant, alpha=0.2, taua=1.0)
    rate = rates + rated
    # plot:
    fig, axs = plt.subplots(2, 1, figsize=(figwidth, 0.6*figwidth))
    ax = axs[0]
    ax.plot(time, stimulus, color=colors['green'], label='stimulus')
    ax.plot(time, deviant, color=colors['orange'], label='deviant', clip_on=False)
    ax.set_ylabel('Stimulus')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.legend(loc='upper left')
    ax = axs[1]
    ax.plot(time, rate, color=colors['blue'], label='adapting')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Spike frequency [Hz]')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    fig.savefig('ssa-ssa')
    

if __name__ == "__main__":
    plot_pulseadaptation()
    plot_sawtoothstimulus()
    plot_cosinestimulus()
    plot_deviant()
    plot_deviantadaptation()
    plot_ssa()
