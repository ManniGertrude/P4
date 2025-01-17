import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.odr as odr

fig, ax = plt.subplots(layout='constrained')
path = os.path.dirname(os.path.abspath(__file__))

files = ['1unbek', '2unbek', '3unbek', '4unbek', 'AgK', 'AgL', 'Au', 'Cu', 'Fe', 'FeZn', 'In', 'Mo', 'Ni', 'Pb', 'Sn', 'Ti', 'W', 'Zn', 'Zr']


def gausfit(func, x, y, beta):
    Low = int(beta[1] - 3 * beta[2])
    High = int(beta[1] + 3 * beta[2])
    x = x[Low:High]
    y = y[Low:High]
    model = odr.Model(func)
    mydata = odr.RealData(x, y, sx=0.005, sy=np.sqrt(y))
    myodr = odr.ODR(mydata, model, beta0=beta)
    out = myodr.run()
    fy = func(out.beta, x)
    chi2 = np.sum((y - fy)**2/np.sqrt(y))/(len(x) - len(out.beta))
    # Printer(f'{out.beta[0]:.5g} $\pm$ {out.sd_beta[0]:.3g} & {out.beta[1]:.5g} $\pm$ {out.sd_beta[1]:.3g} & {out.beta[2]:.5g} $\pm$ {out.sd_beta[2]:.3g} & {out.beta[3]:.5g} $\pm$ {out.sd_beta[3]:.3g} & {out.beta[4]:.5g} $\pm$ {out.sd_beta[4]:.3g} & {chi2:.3g}')
    plt.plot(x, fy,label = f'$\chi^2/ndf = {chi2:.3g} $', alpha = 0.7)
    return out


def Printer(Text):
    Text = Text.replace('.', ',')
    print(Text)


def gauss(Para, x):
    return Para[0] /(Para[2]*np.sqrt(2*np.pi))*np.exp(-0.5*((x - Para[1])/Para[2])**2)


for file in files:
    data = pd.read_csv(f'{path}\\DatenD2\\{file}.txt', sep="\t", header=0, names=['b', 'R'], dtype=str)
    Para = pd.read_csv(f'{path}\\Peaks\\{file}.peaks', sep="\t", header=0, names=['PeakType', 'Center', 'Height', 'Area', 'FWHM', 'Para'])
    Para[['A', 'mu', 'sigma']] = Para['Para'].str.split(expand=True).astype(float)
    data['b'] = data['b'].str.replace(',', '.').astype(float)
    data['R'] = data['R'].str.replace(',', '.').astype(float)
    xData = data['b'].values
    yData = data['R'].values

    for i in range(len(Para)):
        A = Para['A'][i]
        mu = Para['mu'][i]
        sigma = Para['sigma'][i]
        print(A, mu, sigma)
        out = gausfit(gauss, xData, yData, [A, mu, sigma])
        yFit = gauss(out.beta, xData)
        chi2 = np.sum((yData - yFit) ** 2 / np.sqrt(yData)) / (len(xData) - len(out.beta))
    plt.errorbar(xData, yData, xerr=0.05, yerr=np.sqrt(yData), color='darkolivegreen', label='Messwerte', marker='.', zorder=1, linestyle='None')
    plt.grid()
    plt.legend()
    plt.savefig(f'{path}\\Plots\\{file}.pdf')
    plt.cla()
