import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sfa
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors, lighter


def plot_sigmoid():
    fig, axs = plt.subplots(1, 3, figsize=(figwidth, 0.4*figwidth), sharey=True)
    slope = 1.0
    I0 = 0.0
    fmax = 200.0
    I = np.linspace(-2.0, 6.0, 200)
    for ax in axs:
        ax.set_ylim(-1, 200)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(50.0))
    ax = axs[0]
    for k, fmax in enumerate([65.0, 130.0, 200.0]):
        f = sfa.upper_boltzmann(I, fmax, 0.0, 1.0)
        ax.plot(I, f, color=lighter(colors['blue'], (k+1)*0.33), zorder=10, clip_on=False)
    ax.annotate('', (5, 195), (5, 65), zorder=10, arrowprops=dict(arrowstyle='->',
                                                       facecolor='black', lw=0.5))
    ax.text(4, 90.0, r'$f_{\rm max}$', rotation='vertical')
    ax.set_ylabel('Spike frequency [Hz]')
    ax = axs[1]
    for k, I0 in enumerate([2.0, 0.0, -1.0]):
        f = sfa.upper_boltzmann(I, 200.0, I0, 1.0)
        ax.plot(I, f, color=lighter(colors['blue'], (k+1)*0.33), zorder=10, clip_on=False)
    ax.annotate('', (3.3, 100), (0.0, 100), zorder=10, arrowprops=dict(arrowstyle='->',
                                                       facecolor='black', lw=0.5))
    ax.text(1.5, 105.0, r'$I_0$')
    ax.set_xlabel('Stimulus $I$')
    ax = axs[2]
    for k, slope in enumerate([2.0, 1.0, 0.5]):
        f = sfa.upper_boltzmann(I, 200.0, 0.0, slope)
        ax.plot(I, f, color=lighter(colors['blue'], (k+1)*0.33), zorder=10, clip_on=False)
    ax.annotate('', (0.6, 135), (2.8, 115), zorder=10, arrowprops=dict(arrowstyle='->',
                                                       facecolor='black', lw=0.5))
    ax.text(1.8, 127.0, r'$k$')
    fig.savefig('sfa-sigmoid')
    

def plot_stepresponse():
    """ Plot spike frequency and adaptation level in response to step.
    """
    tfac = 1000.0             # plots in milliseconds
    dt = 0.0001               # integration time step in seconds
    time = np.arange(-0.05, 0.3+dt, dt)
    stimulus = np.zeros(len(time)) + 1.0
    stimulus[(time > 0.0) & (time < 0.1)] = 3.0
    rate, adapt = sfa.adaptation_sigmoid(time, stimulus, alpha=0.05)
    fig, (axf, axa) = plt.subplots(2, 1, figsize=(figwidth, 0.6*figwidth), sharex=True)
    axf.plot(tfac*time, rate, color=colors['blue'], zorder=10, clip_on=False)
    axf.set_ylabel('Spike frequency [Hz]')
    axf.xaxis.set_major_locator(ticker.MultipleLocator(100.0))
    axa.plot(tfac*time, adapt, color=colors['red'], clip_on=False)
    axa.set_ylim(0, 3)
    axa.set_xlabel('Time [ms]')
    axa.set_ylabel('Adaptation')
    fig.savefig('sfa-stepresponse')

        
if __name__ == "__main__":
    plot_sigmoid()
    plot_stepresponse()
    
