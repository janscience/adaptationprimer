import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ambiguities as ag
import sys
sys.path.insert(0, '..')
from plotstyle import figwidth, colors


def plot_ambiguities():
    tfac = 1000.0
    # amplitude modulation stimulus for left ear:
    dt = 0.005
    tmax = 1.0
    rng = np.random.RandomState(581)
    am_left = 5.0*(1.0 + 0.3*ag.whitenoise(0.0, 20.0, dt, tmax, rng))
    time = np.arange(len(am_left))*dt
    am_left[time<0.1] = 0.1
    fig, ax = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    ax.plot(tfac*time, am_left)
    ax.set_ylabel('Sound [Pa]')
    ax.set_xlabel('Time [ms]')
    fig.savefig('ambiguities-am')
    # loop through three attenuations of right ear:
    figi, axi = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    figr, axr = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    figs, axs = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    figd, axd = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    figa, axa = plt.subplots(figsize=(figwidth, 0.4*figwidth))
    decibels = [8.0, 4.0, 0.0]
    for k, dbs in enumerate(decibels):
        ampl = 10**(-0.1*dbs)     # attenuation factor from decibels
        am_right = ampl*am_left   # attenuate stimulus for right ear
        # convert to decibels (receptors are logarithmic):
        db_left = 10.0*np.log10(am_left)
        db_right = 10.0*np.log10(am_right)
        db_left[db_left<-10.0] = -10.0
        db_right[db_right<-10.0] = -10.0
        # adapting receptor responses:
        rate_left, _ = ag.adaptation(time, db_left, taua=0.5, alpha=0.4)
        rate_right, _ = ag.adaptation(time, db_right, taua=0.5, alpha=0.4)
        # neuronal noise for left and right receptor:
        rate_left += 10.0*ag.whitenoise(0.0, 60.0, dt, tmax, rng)
        rate_right += 10.0*ag.whitenoise(0.0, 60.0, dt, tmax, rng)
        # sum and difference responses:
        rate_sum = rate_left + rate_right
        rate_diff = rate_left - rate_right
        rate_diff_adapt, _ = ag.adaptation(time, rate_diff, taua=10.0, alpha=50.0, slope=0.01)
        # plot:
        axi.plot(tfac*time, db_right, label='%g dB' % dbs)
        axr.plot(tfac*time, rate_right, label='%g dB' % dbs)
        axs.plot(tfac*time, rate_sum, label='%g dB' % dbs)
        axd.plot(tfac*time, rate_diff, label='%g dB' % dbs)
        axa.plot(tfac*time, rate_diff_adapt, label='%g dB' % dbs)
    axi.set_ylabel('Sound right ear [dB]')
    axi.set_xlabel('Time [ms]')
    axi.legend()
    axr.set_ylabel('Spike frequency [Hz]')
    axr.set_xlabel('Time [ms]')
    axr.legend()
    axs.set_ylabel('Spike frequency [Hz]')
    axs.set_xlabel('Time [ms]')
    axs.legend()
    axd.set_ylabel('Spike frequency [Hz]')
    axd.set_xlabel('Time [ms]')
    axd.legend()
    axa.set_ylabel('Spike frequency [Hz]')
    axa.set_xlabel('Time [ms]')
    axa.legend()
    figi.savefig('ambiguities-dbs')
    figr.savefig('ambiguities-receptors')
    figs.savefig('ambiguities-sum')
    figd.savefig('ambiguities-diff')
    figa.savefig('ambiguities-diffadapt')
    

if __name__ == "__main__":
    plot_ambiguities()
