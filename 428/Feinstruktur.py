import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.odr as odr
from matplotlib.ticker import FixedLocator, FixedFormatter

fig, ax = plt.subplots(layout='constrained')

def Printer(Text):
    Text = Text.replace('.', ',')
    print(Text)

def gauss(Para, x):
    return Para[0] /(Para[2]*np.sqrt(2*np.pi))*np.exp(-0.5*((x - Para[1])/Para[2])**2) + Para[3]*x + Para[4]

def gausfit(func, x, y, farbe, beta):
    model = odr.Model(func)
    mydata = odr.RealData(x, y, sx=0.005, sy=np.sqrt(y)/120)
    myodr = odr.ODR(mydata, model, beta0=beta, maxit=10000)
    out = myodr.run()
    print(f'E in keV: {Deg2Ev(out.beta[1]):.8g}+-{(np.deg2rad(out.sd_beta[1])*0.55151245516*np.cos(np.deg2rad(out.beta[1]/2))/np.sin(np.deg2rad(out.beta[1]/2))**2):.2g}')
    print(f'lambda in pm: {Deg2pm(out.beta[1]):.8g}+-{(np.deg2rad(out.sd_beta[1])*0.55151245516/np.sin(np.deg2rad(out.beta[1]/2))**2):.2g}')
    fy = func(out.beta, x)
    # Printer(f'{out.beta[0]:.5g} $\pm$ {out.sd_beta[0]:.3g} & {out.beta[1]:.5g} $\pm$ {out.sd_beta[1]:.3g} & {out.beta[2]:.5g} $\pm$ {out.sd_beta[2]:.3g} & {out.beta[3]:.5g} $\pm$ {out.sd_beta[3]:.3g} & {out.beta[4]:.5g} $\pm$ {out.sd_beta[4]:.3g} & {np.sum((y - fy)**2/fy)/(len(x) - len(out.beta)):.3g}')
    chi2 = np.sum((y - fy)**2/fy)
    ndf = len(x) - len(out.beta)
    plt.plot(x, fy, c=farbe,label = f'Gauß-Anpassung mit $\chi^2/ndf = {chi2/ndf:.3g} $', alpha=0.7)
    return out

def Deg2pm(x):
    return 2*562.73*np.sin(np.deg2rad(x/2))/1.467

def pm2Deg(x):
    return np.rad2deg(2*np.arcsin(1.467*x/(2*562.73)))

def Deg2Ev(x):
    return 1239.8/Deg2pm(x)

def Ev2Deg(x):
    return pm2Deg(1239.8/x)

path = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(f'{path}\\DatenD2\\Feinstruktur.txt', sep="\t", header=0, names=['b', 'R'], dtype=str)

# Replace commas with dots and convert to float
data['b'] = data['b'].str.replace(',', '.').astype(float)
data['R'] = data['R'].str.replace(',', '.').astype(float)

xData = data['b'].values
yData = data['R'].values

xP1 = xData[150:172]
yP1 = yData[150:172]

xP2 = xData[170:215]
yP2 = yData[170:215]


plt.errorbar(xData, yData, xerr=0.005, yerr=np.sqrt(yData)/120, color='darkolivegreen', label='Messwerte', marker='.', zorder=1, linestyle='None')
# gausfit(gauss, xP2, yP2, 'purple', [426, 20, 0.18, 0, 36.5])
# gausfit(gauss, xP1, yP1, 'blue', [200, 18.25, 0.3, 0, 36.5])

plt.ylabel('Mittlere Zählrate [1/s]', fontsize=12, labelpad=2)
plt.xlabel('Winkel [°]', fontsize=12, labelpad=1)

secax = ax.secondary_xaxis('top', functions=(Deg2pm, pm2Deg))
secax.set_xlabel('Wellenlänge [pm]', fontsize=12, labelpad=3)

# thiax = ax.secondary_xaxis(-0.14)
# thiax.set_xlabel('Energie [keV]', fontsize=12, labelpad=2)

# ticks = np.array([6, 7, 8, 9, 10, 12, 15, 20, 30, 50])
# tick_labels = [Ev2Deg(tick) for tick in ticks]
# thiax.set_xticks(tick_labels )
# thiax.set_xticklabels(ticks)

plt.legend()
plt.grid()
plt.savefig(f'{path}\\Feinstruktur.pdf')

