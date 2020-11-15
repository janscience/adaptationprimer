import matplotlib.pyplot as plt
from cycler import cycler

figwidth = 3.0


""" Muted colors. """
colors_muted = {}
colors_muted['red'] = '#C02717'
colors_muted['orange'] = '#F78017'
colors_muted['yellow'] = '#F0D730'
colors_muted['lightgreen'] = '#AAB71B'
colors_muted['green'] = '#408020'
colors_muted['darkgreen'] = '#007030'
colors_muted['cyan'] = '#40A787'
colors_muted['lightblue'] = '#008797'
colors_muted['blue'] = '#2060A7'
colors_muted['purple'] = '#53379B'
colors_muted['magenta'] = '#873770'
colors_muted['pink'] = '#D03050'
colors_muted['white'] = '#FFFFFF'
colors_muted['gray'] = '#A0A0A0'
colors_muted['black'] = '#000000'

colors = colors_muted
plt.rcParams['axes.prop_cycle'] = cycler(color=[colors['blue'], colors['red'],
                                                colors['lightgreen'], colors['orange'],
                                                colors['cyan'], colors['magenta']])

plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.dpi'] = 200.0
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['font.size'] = 5.0
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 4
plt.rcParams['axes.xmargin'] = 0.0
plt.rcParams['axes.ymargin'] = 0.0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.borderpad'] = 0.0

