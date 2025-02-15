import os 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import r2_score

path = os.path.dirname(os.path.abspath(__file__))

Lambda = [6678.3, 6752.8, 6871.3, 6402.2, 6416.3, 6383.0]

def linear(x, a, b):
    return a * x + b 

def sin(x, a, b):
    return a * np.sin(3.96*x) + b

DateTable = []
Verschiebung = []
VerschiebungError = []

for folder in os.listdir(f'{path}\\BAur'):
    fig, ax = plt.subplots()
    

    DateTable.append((np.datetime64(folder[:10]) - np.datetime64('2021-02-09')) / np.timedelta64(1, 'D'))
    
    Data = pd.read_csv(f'{path}\\BAur\\{folder}\\output.dat', sep='\t', names=['x', 'b', 'y'])[:742]
    # ax.plot(Data['x'], Data['y'], label=folder)

    Maxima = []
    for i in range(6):
        Maxima.append(Data['x'][Data['y'].idxmax()])
        mask = ~((Data['x'] > Maxima[i] - 10) & (Data['x'] < Maxima[i] + 10))
        Data = Data[mask]
        if i == 3:
            # ax.plot(Data['x'][200:], Data['y'][200:], color='green')
            mask = ~((Data['x'] > 200))
            Data = Data[mask]

    Maxima = np.sort(Maxima)
    Lambda = np.sort(Lambda)
    
    # ax.plot(Data['x'], Data['y'], label=folder, color='green')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Kanalnummer')
    # plt.ylabel('Counts')
    # plt.savefig(f'{path}\\BAur\\{folder}\\output.pdf')
    # ax.cla()
    
    popt, pcov = curve_fit(linear, Maxima, Lambda, p0=[1, 1])
    R2score = 1- r2_score(Lambda, linear(np.array(Maxima), *popt))
    # ax.plot(Maxima, linear(np.array(Maxima), *popt), label=f'{popt[0]:.4f} * x + {popt[1]:.0f} mit $R^2$ = 1 - {R2score:.2g}')
    # ax.plot(Maxima, Lambda, 'x', label=f'{folder} (Maxima)', color='crimson')
    # plt.legend()
    # plt.grid()
    # plt.ylabel('Wellenlänge (Angström)')
    # plt.xlabel('Kanalnummer')
    # plt.savefig(f'{path}\\BAur\\{folder}\\fit.pdf')
    plt.close()
    Verschiebung.append(popt[1])
    VerschiebungError.append(np.sqrt(np.diag(pcov))[1])

popt, pcov = curve_fit(sin, DateTable, Verschiebung, p0=[-0.5, 6300], maxfev=10000)
print(popt)
XValues = np.linspace(min(DateTable), max(DateTable), 10000)
plt.plot(XValues, sin(XValues, *popt))
plt.errorbar(DateTable, Verschiebung, yerr=VerschiebungError, fmt='.', color='crimson', capsize=3, linestyle='none')
plt.grid()
plt.xlabel('Datum')
plt.ylabel('Verschiebung (Angström)')
plt.savefig(f'{path}\\Verschiebung.pdf')


