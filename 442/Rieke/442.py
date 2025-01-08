import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.odr as odr
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import csv
import os
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))

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


f = "442welle.csv"

data = read_csv_input(f'{path}\\{f}')


x = np.arange(0, 11.25, 0.05)
 
xdata = np.asarray(data['n'], dtype=np.float64)
yn = np.asarray(data['y_n/cm'], dtype=np.float64)
ynerr = np.asarray(data['Delta y_n/cm'], dtype= np.float64)
x0 = np.asarray(data['x_0/cm'], dtype=np.float64)
xerr = np.asarray(data['Delta x_0/cm'], dtype=np.float64)


ydata = np.cos(yn/x0)
yerr = ((-np.sin(yn/x0)*ynerr/x0)**2 + (-np.sin(yn/x0)*(yn*xerr/(x0**2)))**2)**(1/2)

def fit (x, a, b):
    return a*x- b

def fit_näherung(x,c ,d):
    return c*x + d

yn2= yn**2
yn2err = 2*yn*ynerr

popt, pcov = curve_fit(fit, xdata, ydata, sigma= yerr, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
cov= np.linalg.cond(pcov)
dcov = np.diag(pcov)

ydatapre = fit(xdata, *popt)
res = ydata - ydatapre
chisq = np.sum(((ydata-ydatapre)**2)/yerr**2)
#print ('Residuum ', res)
print ('$\chi^2$=', chisq, '$\chi$/ndf=', chisq/(12-2))

poptn, pcovn = curve_fit(fit_näherung, xdata, yn2, sigma= yn2err, absolute_sigma=True)
perrn = np.sqrt(np.diag(pcovn))
covn= np.linalg.cond(pcovn)
dcovn = np.diag(pcovn)

ydatapren = fit(xdata, *poptn)
resn = yn2 - ydatapren
chisqn = np.sum(((yn2-ydatapren)**2)/yn2err**2)
#print ('Residuum ', res)
print ('$\chi^2$=', chisqn, '$\chi$/ndf=', chisqn/(12-2))

fig, ax= plt.subplots()


plt.grid()
plt.errorbar(xdata, ydata, yerr= yerr, color='midnightblue', marker='+', capsize=2, linestyle='none', label= 'Datenpunkte')
plt.plot(x, fit(x, *popt), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion ')
ax.fill_between(xdata, ydata - yerr, ydata + yerr, alpha=0.2)
ax.set(ylabel=r"cos$ \beta_0 $", xlabel='Ordnung n')

ax.legend()

fig.savefig('wellenlaenge.png')
fig.savefig('wellenlaenge.pdf')
plt.show

fig, ax= plt.subplots()


plt.grid()
plt.errorbar(xdata, yn2, yerr= yn2err, color='midnightblue', marker='+', capsize=2, linestyle='none', label= 'Datenpunkte')
plt.plot(x, fit_näherung(x, *poptn), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion ')
ax.fill_between(xdata, yn2- yn2err, yn2 + yn2err, alpha=0.2)
ax.set(ylabel='Abstandsquadrat $y_n^2$ ', xlabel='Ordnung n')

ax.legend()

fig.savefig('wellenlaenge_näherung.png')
fig.savefig('wellenlaenge_näherung.pdf')
plt.show

print ('fit mit kleinwinkelnäherung', popt, perr)
print (xdata, ydata)
print ('fit mit Laurentreihennäherung:', poptn, perrn)
print(yn)

input_dir = "./"
h = "moden.csv"

data = read_csv_input(os.path.join(input_dir, h))

x = np.arange(0.3, 1, 0.005)

xdata = np.asarray(data['l'], dtype=np.float64)
xerr = np.asarray(data['fehler l'], dtype=np.float64)
ydata = np.asarray(data['fsr'], dtype= np.float64)
yerr= np.asarray(data['fehler fsr'], dtype=np.float64)

print (xdata, ydata)
def fit1(x,a):
    return a/(2*x)

popt, pcov = curve_fit(fit1, xdata, ydata, sigma= yerr, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
cov= np.linalg.cond(pcov)
dcov = np.diag(pcov)

fig, ax= plt.subplots()

plt.grid()
plt.errorbar(xdata, ydata, yerr= yerr, color='midnightblue', marker='+', capsize=2, linestyle='none', label= 'Datenpunkte')
plt.plot(x, fit1(x, *popt), color ='cornflowerblue', linestyle = '-',linewidth='1', label = 'Anpassungsfunktion ')
ax.fill_between(xdata, ydata - yerr, ydata + yerr, alpha=0.2)
ax.set(ylabel=r"$\Delta$ $\nu_{FSR}$ /MHz", xlabel='Resonatorlänge l /m')

ax.legend()

fig.savefig('c.png')
fig.savefig('c.pdf')
plt.show
print (xdata, ydata)
print (popt, perr)

ydatapre = fit1(xdata, *popt)
res = ydata - ydatapre
chisq = np.sum(((ydata-ydatapre)**2)/yerr**2)
#print ('Residuum ', res)
print ('$\chi^2$=', chisq)
