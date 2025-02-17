import os 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import r2_score

path = os.path.dirname(os.path.abspath(__file__))

Lambda = [6383, 6402.2, 6416.3, 6678.3, 6752.8, 6871.3]

def linear(x, a, b):
    return a * x + b 

def sin(x, a, b):
    return a *abs(np.sin(2*np.pi*(x+b)/3.9600400))

DateTable = np.array(['2021-02-09T23:17:09', 
             '2021-02-11T19:28:29',
             '2021-02-12T02:58:07',
             '2021-02-12T19:14:25',
             '2021-02-13T03:10:15',
             '2021-02-13T18:58:06',
             '2021-02-13T23:03:31',
             '2021-02-14T04:04:03',
             '2021-04-23T22:19:53',
             '2021-09-02T00:25:11',
             '2021-09-08T00:28:46',
             '2021-09-09T00:43:39',
             '2021-09-23T00:39:23',
             '2021-10-07T23:04:47',
             '2021-10-28T00:16:03',
             '2021-10-28T23:55:01',
             '2021-11-09T22:24:17',
             '2021-12-20T21:15:02',
             '2021-12-21T00:35:31',
             '2021-12-22T01:14:16'], 
              dtype='datetime64[D]')

ZeitVerschiebung = (DateTable - np.datetime64('2021-03-20T09:37:00')) / np.timedelta64(1, 'D')
ModZeitVerschiebung = ZeitVerschiebung % 3.96004
Verschiebung = []
VerschiebungError = []


Linien = pd.read_csv(f'{path}\\Linien.csv', sep=',', names=['S11', 'S21', 'Fe1', 'Ha1', 'S12', 'S22', 'Fe2', 'Ha2'], skiprows=1)
names1 = ['S11', 'S21', 'Fe1', 'Ha1']
names2 = ['S12', 'S22', 'Fe2', 'Ha2']


delta = np.deg2rad(44.9475)
alpha = 1.56658814
epsilon = np.deg2rad(23)
y = np.cos(delta)*np.cos(alpha)
z = np.cos(epsilon)*np.sin(delta)-np.sin(epsilon)*np.cos(delta)*np.sin(alpha)
B = np.arcsin(z)
L = np.arccos(y/np.cos(B))


Index = -1
for folder in os.listdir(f'{path}\\BAur'):
    Index += 1
    fig, ax = plt.subplots()
        
    Data = pd.read_csv(f'{path}\\BAur\\{folder}\\output.dat', sep='\t', names=['x', 'b', 'y'])[:742]
    # ax.plot(Data['x'], Data['y'], label=f'{folder} (Original)')
    Dataf = Data
    Maxima = []
    for i in range(6):
        Maxima.append(Dataf['x'][Dataf['y'].idxmax()])
        mask = ~((Dataf['x'] > Maxima[i] - 10) & (Dataf['x'] < Maxima[i] + 10))
        Dataf = Dataf[mask]
        if i == 3:
            # ax.plot(Dataf['x'][150:], Dataf['y'][150:], color='green')
            mask = ~((Dataf['x'] > 200))
            Dataf = Dataf[mask]

    Maxima = np.sort(Maxima)
    
    # ax.plot(Dataf['x'], Dataf['y'], label=f'{folder} (Gefiltert)', color='green')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Kanalnummer')
    # plt.ylabel('Counts')
    # plt.savefig(f'{path}\\BAur\\{folder}\\output.pdf')
    # ax.cla()
    
    
    popt, pcov = curve_fit(linear, Maxima, Lambda, p0=[1, 1])
    R2score = 1 - r2_score(Lambda, linear(np.array(Maxima), *popt))
    # ax.plot(Maxima, linear(np.array(Maxima), *popt), label=f'{popt[0]:.4f} * x + {popt[1]:.0f} mit $R^2$ = 1 - {R2score:.2g}')
    # ax.plot(Maxima, Lambda, 'x', label=f'{folder} (Maxima)', color='crimson')
    # plt.legend()
    # plt.grid()
    # plt.ylabel('Wellenlänge (Angström)')
    # plt.xlabel('Kanalnummer')
    # plt.savefig(f'{path}\\BAur\\{folder}\\fit.pdf')
    # ax.cla() 

    ax.errorbar(linear(Data['x'], *popt), Data['b'], yerr=np.sqrt(Data['b']), label=f'{folder}')
    for k in range(4):
        ax.axvline(x=Linien[names1[k]][Index], color = 'red', linestyle='--', alpha=0.5)
        ax.axvline(x=Linien[names2[k]][Index], color = 'green', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('Wellenlänge in Angström', fontsize=12)
    plt.ylabel('Intensität', fontsize=12)
    plt.title(f'Intenistätsspektrum von {folder.replace("-"," ")}', fontsize=14)
    plt.savefig(f'{path}\\BAur\\{folder}\\Spektrum.pdf')
    # plt.show()
    ax.cla()




# Linie = [0,0,0,0] 
# for j in range(4):
#     Linie[j] = (np.sum(Linien[names1[j]])+np.sum(Linien[names2[j]]))/(len(Linien[names1[j]]) + len(Linien[names2[j]]))
Linie = [6346, 6370.6, 6456, 6562.5]
LE = 2*np.pi*ZeitVerschiebung/365.25
vErde = -29800*np.cos(B)*np.sin(LE-L)
XValues = np.linspace(min(ModZeitVerschiebung), max(ModZeitVerschiebung), 1000)
Colortable = ['crimson', 'blue', 'green', 'purple']
for i in range(4):
    Verschiebung = abs((Linien[names1[i]]-Linie[i])*3e8/Linie[i]  - ((Linien[names2[i]]-Linie[i])*3e8/Linie[i]))
    VerschiebungErr = np.sqrt((0.1*3e8/Linie[i])**2 + (0.1*(Linien[names1[i]]-Linie[i])*3e8/Linie[i]**2)**2)
    plt.errorbar(ModZeitVerschiebung, Verschiebung, yerr=VerschiebungErr, elinewidth=0.1, linestyle='none', label = names1[i][:-1], color=Colortable[i], marker='x')
    popt, pcov = curve_fit(sin, ModZeitVerschiebung, Verschiebung, sigma=VerschiebungErr, absolute_sigma=True, p0=[1, 1], maxfev=10000)
    plt.plot(XValues, sin(XValues, *popt), color = Colortable[i],)
    plt.grid()
    plt.legend()
    plt.xlabel('Zeit in Tagen')
    plt.ylabel('Geschwindigkeit zueinander km/s')
    plt.savefig(f'{path}\\Verschiebung{names1[i][:-1]}.pdf')
    ax.cla()
    chi2red = np.sum(((Verschiebung - sin(ModZeitVerschiebung, *popt))**2)/VerschiebungErr**2)/(len(ModZeitVerschiebung)-2)
    print(popt, np.sqrt(np.diag(pcov)))
    print(chi2red)


