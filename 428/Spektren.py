import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

fig, ax = plt.subplots(layout='constrained')
path = os.path.dirname(os.path.abspath(__file__))

files = ['1unbek', '2unbek', '3unbek', '4unbek', 'AgK', 'AgL', 'Au', 'Cu', 'Fe', 'FeZn', 'In', 'Mo', 'Ni', 'Pb', 'Sn', 'Ti', 'W', 'Zn', 'Zr']

def gauss(x, *p):
    y = np.zeros_like(x)
    for i in range(0, len(p), 3):
        A = p[i]
        mu = p[i+1]
        sigma = p[i+2]
        y = y + A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y


def Printer(Text):
    Text = Text.replace('.', ',')
    print(Text)


for i in range(len(files)):
    data = pd.read_csv(f'{path}\\DatenD2\\{files[i]}.txt', sep="\t", header=0, names=['b', 'R'], dtype=str)
    Para = pd.read_csv(f'{path}\\Peaks\\{files[i]}.peaks', sep="\t", header=0, names=['PeakType', 'Center', 'Height', 'Area', 'FWHM', 'Para'])
    Params = Para['Para'].str.split(expand=True).astype(float).values.flatten()
    data['b'] = data['b'].str.replace(',', '.').astype(float)
    data['R'] = data['R'].str.replace(',', '.').astype(float)
    xData = data['b'].values
    yData = data['R'].values

    popt, pcov = curve_fit(gauss, xdata=xData, ydata=yData, p0=Params)
    fit = gauss(xData, *popt)
    poptErr = np.sqrt(np.diag(pcov))
    chi2 = np.sum((yData - fit)**2 / np.sqrt(yData+1e-5))/(len(yData) - len(Params))
    print(f'\hline \hline \n {files[i]} & {chi2} &  & \\\\ \n \hline')
    for j in range(0, len(popt), 3):
        if i*100+j in [6, 106, 203, 306, 309, 415, 418, 506, 512, 612, 703, 803, 806, 809, 906, 903, 1118, 1124, 1203, 1312, 1403, 1606, 1703]:
            Printer(f'({j//3+1}) & {popt[j]:.0f} $\pm$ {poptErr[j]:.2g} & {popt[j+1]:.3f} $\pm$ {poptErr[j+1]:.2g} & {popt[j+2]:.4f} $\pm$ {poptErr[j+2]:.2g}\\\\')
            plt.plot(xData, gauss(xData, *popt[j:j+3]), label=f'Gauss ({j//3+1})', zorder=30, alpha=0.5, linestyle='--')

        else:
            Printer(f'{j//3+1} & {popt[j]:.0f} $\pm$ {poptErr[j]:.2g} & {popt[j+1]:.3f} $\pm$ {poptErr[j+1]:.2g} & {popt[j+2]:.4f} $\pm$ {poptErr[j+2]:.2g}\\\\')
            plt.plot(xData, gauss(xData, *popt[j:j+3]), label=f'Gauss {j//3+1}', zorder=30, alpha=0.7, linestyle='--')
    plt.plot(xData, fit, color='darkred', label=f'$\chi^2/ndf$ = {chi2:.2f}', zorder=2)
    plt.errorbar(xData, yData, xerr=0.05, yerr=np.sqrt(yData), color='darkolivegreen', label='Messwerte', marker='.', zorder=1, linestyle='None')
    plt.grid()
    plt.legend()
    plt.savefig(f'{path}\\Plots\\{files[i]}.pdf')
    plt.cla()
