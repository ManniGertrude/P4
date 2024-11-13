import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import csv
import os
import pandas as pd



def read_csv_input(test):
    with open(test, "r") as file:
        dictR = csv.DictReader(file, delimiter=",")
        data = {hdr: [] for hdr in dictR.fieldnames}
        for row in dictR:
            for field in row:
                try:
                    data[field].append(float(row[field]))
                except ValueError:
                    data[field].append(0.)
    return data

input_dir = "./"
f = "gegen1letzte3.csv"

data = read_csv_input(os.path.join(input_dir, f))

def gaus(x,a, o,m,a1,o1,m1,a2,o2,m2,a3,o3,m3):
    g1= a/(o*(2*np.pi)**(1/2))*np.exp(-(x-m)**2/(2*o)**2)
    g2= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-m1)**2/(2*o1)**2) 
    g3= a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-m2)**2/(2*o2)**2)
    g4= a3*np.exp(-(x-m3)*o3)
    #lin = d*x + n
    return g1 +g2 +g3+ g4 

def fit_multi_gaus(data):
    y_err = [0.01*value for value in data['U_A / V']]
    xdata = np.asarray(data['U_B / V'], dtype=np.float64)
    ydata = np.asarray(data['U_A / V'], dtype=np.float64)
    yerr = np.asarray(y_err)
    plt.grid()
    plt.errorbar(xdata, ydata, color='cornflowerblue', yerr= yerr, marker='o', linestyle='none')
    p0 = np.asarray([10, 26, 2,10,31,2, 10,36, 2,10, 40,2 ])
    popt, pcov = curve_fit(gaus, xdata, ydata,  p0=p0, sigma=yerr,absolute_sigma=True, maxfev=int(1e6))
    perr = np.sqrt(np.diag(pcov))
    print (popt,perr)
    plt.plot(xdata, gaus(xdata, *popt), color ='midnightblue', linestyle = '-', label='fit')
    bottom, top = plt.ylim()
    plt.ylim(bottom, top*1.2)
    plt.xlabel('Beschleunigungsspannung $U_{B}$ /V')
    plt.ylabel('Anodenstrom $I_{A}$ /A')
    plt.legend()
    plt.savefig('gausmulti3.png')
    plt.show


fit_multi_gaus(data)


xdata = np.asarray(data['UB/V'], dtype=np.float64)
ydata = np.asarray(data['UA/V'], dtype=np.float64)
popt, pcov = curve_fit(gaus, xdata, ydata, maxfev = 100000000)
perr = np.sqrt(np.diag(pcov))
cov= np.linalg.cond(pcov)
dcov = np.diag(pcov)



fig, ax=plt.subplots()
plt.grid()
plt.plot(xdata, ydata, color='cornflowerblue', marker='o', linestyle='none', label = 'Datenpunkte')
#plt.plot(xdata, gaus(xdata, *popt), color ='midnightblue', linestyle = '-')
ax.set(ylabel='Driftstrom $I_{A}$ /A', xlabel='Beschleunigungsspannung $U_{B}$ /V')

ax.legend()

fig.savefig('gausgegen1.png')
plt.show