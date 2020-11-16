import matplotlib.pyplot as plt
from cycler import cycler
try:
    from matplotlib.colors import colorConverter as cc
except ImportError:
    import matplotlib.colors as cc
try:
    from matplotlib.colors import to_hex
except ImportError:
    from matplotlib.colors import rgb2hex as to_hex

    
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


def lighter(color, lightness):
    """ Make a color lighter.

    Parameters
    ----------
    color: dict or matplotlib color spec
        A matplotlib color (hex string, name color string, rgb tuple)
        or a dictionary with an 'color' or 'facecolor' key.
    lightness: float
        The smaller the lightness, the lighter the returned color.
        A lightness of 0 returns white.
        A lightness of 1 leaves the color untouched.
        A lightness of 2 returns black.

    Returns
    -------
    color: string or dict
        The lighter color as a hexadecimal RGB string (e.g. '#rrggbb').
        If `color` is a dictionary, a copy of the dictionary is returned
        with the value of 'color' or 'facecolor' set to the lighter color.
    """
    try:
        c = color['color']
        cd = dict(**color)
        cd['color'] = lighter(c, lightness)
        return cd
    except (KeyError, TypeError):
        try:
            c = color['facecolor']
            cd = dict(**color)
            cd['facecolor'] = lighter(c, lightness)
            return cd
        except (KeyError, TypeError):
            if lightness > 2:
                lightness = 2
            if lightness > 1:
                return darker(color, 2.0-lightness)
            if lightness < 0:
                lightness = 0
            r, g, b = cc.to_rgb(color)
            rl = r + (1.0-lightness)*(1.0 - r)
            gl = g + (1.0-lightness)*(1.0 - g)
            bl = b + (1.0-lightness)*(1.0 - b)
            return to_hex((rl, gl, bl)).upper()

