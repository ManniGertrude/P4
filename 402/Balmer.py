import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import os


# Systempfad
input_dir = os.path.dirname(os.path.abspath(__file__))


# Einlesen der Daten
def read_csv_input(test, head, skip):
    data = pd.read_csv(f'{input_dir}\\{test}', sep="\t", header=head, skiprows=skip).astype(np.float64)
    return data

# 1 Lorenzfunktionen
def l1(x, h, x0, a, c):
    return  h/(1+((x-x0)*2/a)**2) + c 

def l2(x, h1, x1, a1, c1, h2, x2, a2, c2):
    Summe = l1(x, h1, x1, a1, 0) + l1(x, h2, x2, a2, 0) + (c1 + c2)/2
    return Summe



# Farbpalette
ColorTable = ['crimson','navy','olive', 'green', 'darkcyan',
              'slateblue', 'orchid', 'deeppink', 'purple', 'black']

# Dateinamen
e = ['Halpha.txt', 'Hbeta.txt', 'Hgamma.txt']


# Fitparameter
p0 = [[71, -0.065,  0.012, 6, -0.033, -0.015, 22],
      [28,  0.022, 0.006, 1.5, 0.042, 0.0065, 20],
      [10, 0, 0.02, 0.9, -0.05, 0.01, 23]]

lower_bounds = [[20, -0.07,  0, 0, -0.1, -0.05, 15],
                [10,  0.001, 0.001, 1, 0.0, 0, 15],
                [1, -0.09, 0, 0.9, -0.05, 0.01, 15]]

upper_bounds = [[80, -0.05,  0.1, 20, 0, 0, 30],
                [50,  0.1, 0.01, 10, 0.1, 0.1, 30],
                [100, -0.07, 0.1, 0.9, -0.05, 0.01, 30]]

Grenze = [[-0.12, -0.01, -0.06],
        [-0.05, 0.031, 0.008],
        [0, 0.06, 0.05]]

# Fit
for i in range(len(e)):
    data = read_csv_input(f'Daten\\Balmer\\{e[i]}', 0, 0)
    xValues = np.linspace(Grenze[0][i], Grenze[2][i], 100)
    x = np.asarray(data['a'], dtype=np.float64)
    y = np.asarray(data['I'], dtype=np.float64)

    xdata = x[np.where((x > Grenze[0][i]) & (x < Grenze[2][i]))]
    ydata = y[np.where((x > Grenze[0][i]) & (x < Grenze[2][i]))]
    xErr = [0.001]*len(xdata)
    yErr = [0.1]*len(ydata)
    popt = []
    for j in range(2):
        xP = xdata[np.where((xdata > Grenze[j][i]) & (xdata < Grenze[j+1][i]))]
        yP = ydata[np.where((xdata > Grenze[j][i]) & (xdata < Grenze[j+1][i]))]
        p, c = curve_fit(l1, xP, yP, p0=[*p0[i][3*j+0:3*j+3],p0[i][6]], maxfev = 100000)
        popt.append([p[0], p[1], p[2], p[3]])
        ydatapre = l1(xP, *popt[j])
        chisq = np.sum(((yP-ydatapre)**2)/(abs(yP)*0.01)**2)
        chi_ndf = chisq/(len(xP)-4)
        plt.plot(xValues, l1(xValues, *popt[j]), label = 'Anpassung 1', zorder = 10, alpha = 0.8, c = ColorTable[j])
        # print(f'\chi^2{e[i][:-4]}: {chi_ndf:.2f}')
    plt.xlim(Grenze[0][i], Grenze[2][i])
    plt.errorbar(x=xdata, y=ydata, xerr=xErr, yerr=yErr, label='Datenpunkte', color='green', zorder = 0)
    if i == 2:
        poptTotal, cov = curve_fit(l2, xdata, ydata, p0=[*popt[0], *popt[1]], maxfev = 100000)
        plt.plot(xValues, l2(xValues, *poptTotal), label = 'Summe', zorder = 10, alpha = 0.8, c = 'black')
    plt.grid()
    plt.legend()
    plt.savefig(f'{input_dir}\\Plots\\Balmer\\{e[i][:-4]}.pdf')
    plt.cla()