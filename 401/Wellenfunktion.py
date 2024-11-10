from scipy.constants import physical_constants
import matplotlib.pyplot as plt
import scipy.special as sp
import seaborn as sns
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))


def radialfunc(n, l, r, a0):
    laguerre = sp.genlaguerre(n - l - 1, 2 * l + 1)
    p = 2 * r / (n * a0)
    constant_factor = np.sqrt(((2 / n * a0) ** 3 * (sp.factorial(n - l - 1)))/(2 * n * (sp.factorial(n + l))))
    return constant_factor * np.exp(-p / 2) * (p ** l) * laguerre(p)


def winkelfunc(m, l, theta, phi):
    legendre = sp.lpmv(m, l, np.cos(theta))
    constant_factor = ((-1) ** m) * np.sqrt(((2 * l + 1) * sp.factorial(l - np.abs(m)))/(4 * np.pi * sp.factorial(l + np.abs(m))))
    return constant_factor * legendre * np.real(np.exp(1.j * m * phi))


def WF(n, l, m, a0_scale_factor):
    a0 = a0_scale_factor * physical_constants['Bohr radius'][0] * 1e+12
    grid_extent = 480
    grid_resolution = 680
    z = x = np.linspace(-grid_extent, grid_extent, grid_resolution)
    z, x = np.meshgrid(z, x)
    eps = np.finfo(float).eps
    psi = radialfunc(n, l, np.sqrt((x ** 2 + z ** 2)), a0) * winkelfunc(m, l, np.arctan(x / (z + eps)), 0)
    return psi


def WD(n, l, m, a0_scale_factor, colormap='rocket'):
    
    sns.color_palette(colormap)
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.major.width'] = 4
    plt.rcParams['ytick.major.width'] = 4
    plt.rcParams['xtick.major.size'] = 15
    plt.rcParams['ytick.major.size'] = 15
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['axes.linewidth'] = 4

    fig, ax = plt.subplots(figsize=(16, 16.5))
    plt.subplots_adjust(top=0.82)
    plt.subplots_adjust(right=0.905)
    plt.subplots_adjust(left=-0.1)


    psi = WF(n, l, m, a0_scale_factor)
    prob_density = np.abs(psi) ** 2
    im = ax.imshow(np.sqrt(prob_density).T, cmap=sns.color_palette(colormap, as_cmap=True))
    cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
    cbar.set_ticks([])
    
    
    background_color = sorted(sns.color_palette(colormap, n_colors=100),key=lambda color: 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])[0]
    plt.rcParams['text.color'] = '#dfdfdf'
    title_color = '#dfdfdf'
    fig.patch.set_facecolor(background_color)
    cbar.outline.set_visible(False)
    ax.tick_params(axis='x', colors='#c4c4c4')
    ax.tick_params(axis='y', colors='#c4c4c4')
    for spine in ax.spines.values():
        spine.set_color('#c4c4c4')


    ax.set_title('Hydrogen Atom - Wavefunction Electron Density', pad=130, fontsize=44, loc='left', color=title_color)
    ax.text(0, 722, (r'$|\psi_{n \ell m}(r, \theta, \varphi)|^{2} ='r' |R_{n\ell}(r) Y_{\ell}^{m}(\theta, \varphi)|^2$'), fontsize=36)
    ax.text(30, 615, r'$({0}, {1}, {2})$'.format(n, l, m), color='#dfdfdf', fontsize=42)
    ax.text(770, 140, 'Electron probability distribution', rotation='vertical', fontsize=40)
    ax.text(705, 700, 'Higher\nprobability', fontsize=24)
    ax.text(705, -60, 'Lower\nprobability', fontsize=24)
    ax.text(775, 590, '+', fontsize=34)
    ax.text(769, 82, '−', fontsize=34, rotation='vertical')
    ax.invert_yaxis()
    plt.savefig(f'{path}\\Wellenfunktionen\\({n},{l},{m}).png')


for n in range (1, 4):
    for l in range(0, n):
        for m in range(-l, l + 1):
            WD(n, l, m, 0.2, colormap='magma')