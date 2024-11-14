import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import numpy as np
import pandas as pd
import os

input_dir = os.path.dirname(os.path.abspath(__file__))


def read_csv_input(test, head, skip):
    data = pd.read_csv(f'{input_dir}\\{test}', sep="\t", header=head, skiprows=skip).astype(np.float64)
    return data


def objective(params, x, y):
    residuals = g3(x, *params) - y
    return np.sum(residuals**2)


def polyfit(x, a, b, c,  d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x**1 + e


def g3(x, a1, x1, o1, a2, x2, o2, a3, x3, o3, c, m):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-x1)**2/(2*o1)**2)
    g2 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-x2)**2/(2*o2)**2)
    g3 = a3/(o3*(2*np.pi)**(1/2))*np.exp(-(x-x3)**2/(2*o3)**2)
    return g1 + g2 + g3 - m*(x)**2 + c


e = ['Data\\Strom zu Magnetfeld danach.csv', 
     'Data\\Strom zu Magnetfeld davor.csv']
e_Names = ['danach', 'davor']


f = ["Data\\Intensitäten\\0.csv",
     "Data\\Intensitäten\\1.csv", 
     "Data\\Intensitäten\\2.csv", 
     "Data\\Intensitäten\\3.csv", 
     "Data\\Intensitäten\\4.csv",
     "Data\\Intensitäten\\5.csv", 
     "Data\\Intensitäten\\6.csv", 
     "Data\\Intensitäten\\7.csv",
     "Data\\Intensitäten\\8.csv", 
     "Data\\Intensitäten\\9.csv"]


ColorTable = ['crimson','darkgoldenrod','olive', 'green', 'darkcyan',
              'slateblue', 'orchid', 'deeppink', 'purple', 'black']


# Ampere zu Tesla 
popts = []
for i in range(len(e)):
    data = read_csv_input(e[i], 3, 2)
    ydata = np.asarray(data['B'], dtype=np.float64)
    xdata = np.asarray(data['I'], dtype=np.float64)
    yerr = np.asarray(0.01*abs(data['B']) + 0.005, dtype=np.float64)
    popt, pcov = curve_fit(polyfit, xdata, ydata, p0=[1, 1, 1, 1, 1], sigma=yerr, absolute_sigma=True)
    popts.append(popt)
    perr = np.sqrt(np.diag(pcov))
    ydatapre = polyfit(xdata, *popt)
    chisq = np.sum(((ydata-ydatapre)**2)/yerr**2)
    chi_ndf = chisq/(len(xdata)-5)
    Print = f'{popt[0]:.3f} $\pm$ {perr[0]:.3f} & {popt[1]:.3f} $\pm$ {perr[1]:.3f} & {popt[2]:.3f} $\pm$ {perr[2]:.3f} & {popt[3]:.3f} $\pm$ {perr[3]:.3f} & {popt[4]:.3f} $\pm$ {perr[4]:.3f} & {chisq:.2f} & {chi_ndf:.2f} \\\\'
    Print = Print.replace('.', ',')
    # print(e_Names[i])
    # print(Print)
    plt.plot(xdata, polyfit(xdata, *popt), color = ColorTable[i],
            label=f'Polynom {e_Names[i]}', zorder = 10)
    plt.errorbar(xdata, ydata, yerr= yerr, color=ColorTable[i+4], marker='o',markersize =1, 
                     linestyle='none', label=f'Datenpunkte {e_Names[i]}', zorder = 1, alpha = 0.8)
yDiff = polyfit(xdata, *popts[0]) - polyfit(xdata, *popts[1])
yValues = 0.5*polyfit(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), *popts[0])+ 0.5*polyfit(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), *popts[1])
plt.plot(xdata, yDiff, color = ColorTable[3],
            label=f'Differenz der Polynome ', zorder = 1)
plt.plot(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), yValues, color = ColorTable[9],
            label=f'Mittel der Polynome ', zorder = 1)
plt.grid()
plt.ylabel('Magnetfeld $B$ / mT')
plt.xlabel('Stromstärke $I$ / A')
plt.title('Magnetfeld in Abhängigkeit des Stroms')
plt.legend()
plt.savefig(f'{input_dir}\\Output\\ZM\\BFeld.pdf')
plt.cla()


# Intensitäten einzeln
for i in range(len(f)):
    data = read_csv_input(f[i], 0, 0)
    xdata = np.asarray(data['a'], dtype=np.float64)
    ydata = np.asarray(data['I1'], dtype=np.float64)
    plt.plot(xdata, ydata, color=ColorTable[i], label=f'{f[i][-5]}')
    plt.title(f'Intensität in Abhängigkeit des Winkels für {yValues[i]:.0f} mT')
    plt.grid()
    plt.ylabel('Intensität $I$ /A')
    plt.xlabel('Winkel $\\alpha$ /°')
    plt.savefig(f'{input_dir}\\Output\\ZM\\{f[i][-5]}A.pdf')
    plt.cla()
    
# Intensitäten zusammen
for i in range(len(f)):
    data = read_csv_input(f[i], 0, 0)
    xdata = np.asarray(data['a'], dtype=np.float64)
    ydata = np.asarray(data['I1'], dtype=np.float64)
    plt.plot(xdata, ydata, color=ColorTable[i], label=f'{yValues[i]:.0f} mT')
plt.legend()
plt.title(f'Intensität in Abhängigkeit des Winkels für verschiedene Magnetfelder')
plt.grid()
plt.ylabel('Intensität $I$ /A')
plt.xlabel('Winkel $\\alpha$ /°')
plt.savefig(f'{input_dir}\\Output\\ZM\\Alle.pdf')
plt.cla()


# bounds        a1,    x1,  o1,  a2,    x2,   o2,   a3,    x3,   o3,    c,  m
p0           = [[1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [0.9, -0.74, 0.02, 0.8, -0.67, 0.02,  4.3, -0.6, 0.05, 16.8,1.2],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0,-0.58, 0.04,   15,  0],
                [1.3, -0.75, 0.03, 1.5, -0.67, 0.02,  4.5,-0.56, 0.05,   20,1.3],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.8, 0.03, 5.0, -0.68, 0.02,  4.0, -0.6, 0.04,   15,  0]]
  
  
  
lower_bounds = [[  0, -0.85,    0,   0,  -0.7,    0,    0,-0.65,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0,-0.65,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0,-0.65,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0,-0.65,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6, 0.04,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0,-0.56,    0,   18,   0],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6,    0,  -10, -10]]

upper_bounds = [[10,-0.65, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.40, 0.1,  30,   2],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [ 2,-0.78, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20]]

# einzelner Peak
GrenzeUnten = -0.9
GrenzeOben = -0.3
for i in range(len(f)):
    data = read_csv_input(f[i], 0, 0)
    xdata = np.asarray(data['a'], dtype=np.float64)
    ydata = np.asarray(data['I1'], dtype=np.float64)
    xdata_f = xdata[np.where((xdata > GrenzeUnten) & (xdata < GrenzeOben))]
    ydata_f = ydata[np.where((xdata > GrenzeUnten) & (xdata < GrenzeOben))]
    plt.errorbar(xdata_f, ydata_f, xerr=abs(xdata_f)*0.01, yerr=abs(ydata_f)*0.01, color=ColorTable[9], label=f'{yValues[i]:.0f} mT')
    popt, pcov = curve_fit(g3, xdata_f, ydata_f, p0=p0[i], bounds=(lower_bounds[i], upper_bounds[i]), maxfev=1e9)
    perr = np.sqrt(np.diag(pcov))
    ydatapre = g3(xdata_f, *popt)
    chisq = np.sum(((ydata_f-ydatapre)**2)/(abs(ydata_f)*0.01)**2)
    chi_ndf = chisq/(len(xdata_f)-5)
    Print = f'{popt[0]:.3f} $\pm$ {perr[0]:.3f} & {popt[1]:.3f} $\pm$ {perr[1]:.3f} & {popt[2]:.3f} $\pm$ {perr[2]:.3f} & {popt[3]:.3f} $\pm$ {perr[3]:.3f} & {popt[4]:.3f} $\pm$ {perr[4]:.3f} & {popt[5]:.3f} $\pm$ {perr[5]:.3f} & {popt[6]:.3f} $\pm$ {perr[6]:.3f} & {popt[7]:.3f} $\pm$ {perr[7]:.3f} & {popt[8]:.3f} $\pm$ {perr[8]:.3f} & {popt[9]:.3f} $\pm$ {perr[9]:.3f} & {popt[10]:.3f} $\pm$ {perr[10]:.3f} & {chisq:.3f} & {chi_ndf:.3f} \\\\'
    Print = Print.replace('.', ',')
    print(Print)
    plt.plot(xdata_f, g3(xdata_f, *popt), color = 'red',linestyle ='-', label =f'Anpassungskurve', zorder = 10, alpha = 0.7) 
    plt.legend( loc='upper right')
    plt.title(f'Intensität in Abhängigkeit des Winkels für verschiedene Magnetfelder')
    plt.grid()
    plt.ylabel('Intensität $I$ /A')
    plt.xlabel('Winkel $\\alpha$ /°')
    plt.savefig(f'{input_dir}\\Output\\ZM\\Peak_{f[i][-5]}A.pdf')
    plt.cla()