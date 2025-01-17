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
    mydata = odr.RealData(x, y, sx=0.005, sy=np.sqrt(120*y)/120)
    myodr = odr.ODR(mydata, model, beta0=beta, maxit=10000)
    out = myodr.run()
    # print(f'Winkel {out.beta[1]} $\pm$ {out.sd_beta[1]}')
    # print(f'E in keV: {Deg2Ev(out.beta[1]):.8g}$\pm$ {(np.deg2rad(out.sd_beta[1])*0.55151245516*np.cos(np.deg2rad(out.beta[1]/2))/np.sin(np.deg2rad(out.beta[1]/2))**2):.2g}')
    # print(f'lambda in pm: {Deg2pm(out.beta[1]):.8g}$\pm$ {(np.deg2rad(out.sd_beta[1])*0.55151245516/np.sin(np.deg2rad(out.beta[1]/2))**2):.2g}')
    fy = func(out.beta, x)
    chi2 = np.sum((y - fy)**2 *120/np.sqrt(y))/(len(x) - len(out.beta))
    # Printer(f'{out.beta[0]:.5g} $\pm$ {out.sd_beta[0]:.3g} & {out.beta[1]:.5g} $\pm$ {out.sd_beta[1]:.3g} & {out.beta[2]:.5g} $\pm$ {out.sd_beta[2]:.3g} & {out.beta[3]:.5g} $\pm$ {out.sd_beta[3]:.3g} & {out.beta[4]:.5g} $\pm$ {out.sd_beta[4]:.3g} & {chi2:.3g}')
    plt.plot(x, fy, c=farbe,label = f'Gauß-Anpassung mit $\chi^2/ndf = {chi2:.3g} $', alpha = 0.7)
    return out

def Deg2pm(x):
    return 2*562.73*np.sin(np.deg2rad(x/2))/4

def pm2Deg(x):
    return np.rad2deg(2*np.arcsin(4*x/(2*562.73)))

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

xP1 = xData[75:150]
yP1 = yData[75:150]

xP2 = xData[280:340]
yP2 = yData[280:340]


plt.errorbar(xData, yData, xerr=0.005, yerr=np.sqrt(120*yData)/120, color='darkolivegreen', label='Messwerte', marker='.', zorder=1, linestyle='None')
gausfit(gauss, xP1, yP1, 'blue', [0.6, 29.6, 0.05, 0, 0.3])
plt.ylabel('Mittlere Zählrate [1/s]', fontsize=12, labelpad=2)
plt.xlabel('Winkel [°]', fontsize=12, labelpad=1)
secax = ax.secondary_xaxis('top', functions=(Deg2pm, pm2Deg))
secax.set_xlabel('Wellenlänge [pm]', fontsize=12, labelpad=3)

ax.axvline(pm2Deg(70.93+0.75), label = 'Literaturwerte')
ax.axvline(pm2Deg(71.36+0.75))
plt.legend(loc = 'upper left', framealpha=0.3)

plt.grid()
ax.set_xlim(29.4, 29.8)
plt.savefig(f'{path}\\FeinstrukturZoom.pdf')
plt.cla()


ax.axvline(pm2Deg(70.93), label = 'Literaturwerte')
ax.axvline(pm2Deg(71.36))

plt.errorbar(xData, yData, xerr=0.005, yerr=np.sqrt(120*yData)/120, color='darkolivegreen', label='Messwerte', marker='.', zorder=1, linestyle='None')
gausfit(gauss, xP1, yP1, 'blue', [0.6, 29.6, 0.05, 0, 0.3])
gausfit(gauss, xP2, yP2, 'crimson', [0.15, 31.6, 0.08, 0, 0.3])
plt.ylabel('Mittlere Zählrate [1/s]', fontsize=12, labelpad=2)
plt.xlabel('Winkel [°]', fontsize=12, labelpad=1)
secax = ax.secondary_xaxis('top', functions=(Deg2pm, pm2Deg))
secax.set_xlabel('Wellenlänge [pm]', fontsize=12, labelpad=3)

thiax = ax.secondary_xaxis(-0.14)
thiax.set_xlabel('Energie [keV]', fontsize=12, labelpad=2)

ticks = np.array([15, 15.5, 16, 16.5, 17, 17.5, 18])
tick_labels = [Ev2Deg(tick) for tick in ticks]
thiax.set_xticks(tick_labels )
thiax.set_xticklabels(ticks)
plt.legend(loc = 'upper right', framealpha=0.8)
plt.grid()
ax.set_xlim(28.45, 32.05)
plt.savefig(f'{path}\\Feinstruktur.pdf')


