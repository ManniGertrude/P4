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


def lin(x, a, b):
    return a*x + b

def polyfit(x, a, b, c,  d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x**1 + e


def g3(x, a1, x1, o1, a2, x2, o2, a3, x3, o3, c, m):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-x1)**2/(2*o1)**2)
    g2 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-x2)**2/(2*o2)**2)
    g3 = a3/(o3*(2*np.pi)**(1/2))*np.exp(-(x-x3)**2/(2*o3)**2)
    return g1 + g2 + g3 - m*(x)**2 + c


def g1(x, A, mu, sigma, c):
    return A/(sigma*(2*np.pi)**(1/2))*np.exp(-(x-mu)**2/(2*sigma)**2) + c


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
perrs = []
for i in range(len(e)):
    data = read_csv_input(e[i], 3, 2)
    ydata = np.asarray(data['B'], dtype=np.float64)
    xdata = np.asarray(data['I'], dtype=np.float64)
    yerr = np.asarray(0.02*abs(data['B'])+0.005, dtype=np.float64)
    popt, pcov = curve_fit(polyfit, xdata, ydata, p0=[1, 1, 10, 1, 0.001], sigma=yerr, absolute_sigma=True, maxfev=10000)
    popts.append(popt)
    perr = np.sqrt(np.diag(pcov))
    perrs.append(perr)
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
# Print1 = f' Davor & {popts[1][0]:.4f} $\pm$ {perrs[1][0]:.4f} & {popts[1][1]:.3f} $\pm$ {perrs[1][1]:.3f} & {popts[1][2]:.2f} $\pm$ {perrs[1][2]:.2f} & {popts[1][3]:.2f} $\pm$ {perrs[1][3]:.2f} & {popts[1][4]:.4f} $\pm$ {perrs[1][4]:.4f} \\\\'
# Print2 = f' Danach & {popts[0][0]:.4f} $\pm$ {perrs[0][0]:.4f} & {popts[0][1]:.3f} $\pm$ {perrs[0][1]:.3f} & {popts[0][2]:.2f} $\pm$ {perrs[0][2]:.2f} & {popts[0][3]:.2f} $\pm$ {perrs[0][3]:.2f} & {popts[0][4]:.4f} $\pm$ {perrs[0][4]:.4f} \\\\'
# Print1 = Print1.replace('.', ',')
# Print2 = Print2.replace('.', ',')
# print(Print1)
# print(Print2)
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
                [1.5,  -0.7, 0.01, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [0.9, -0.74, 0.02, 0.8, -0.67, 0.02,  4.3, -0.6, 0.02, 16.8,1.2],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0,-0.58, 0.04,   15,  0],
                [1.3, -0.75, 0.03, 1.5, -0.67, 0.02,  4.5,-0.56, 0.05,   20,1.3],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.7, 0.03, 1.0, -0.66, 0.02,  4.0, -0.6, 0.04,   15,  0],
                [1.5,  -0.8, 0.03, 5.0, -0.68, 0.02,  4.0, -0.6, 0.04,   15,  0]]
  
  
  
lower_bounds = [[  0, -0.85,    0,   0,  -0.7,    0,    0,-0.65,    0,  -10, -10],
                [  0, -0.9,    0,   0,  -0.7,    0,    0,-0.65,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0,-0.65,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0,-0.65,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6, 0.04,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0,-0.56,    0,   18,   0],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6,    0,  -10, -10],
                [  0, -0.85,    0,   0,  -0.7,    0,    0, -0.6,    0,  -10, -10]]

upper_bounds = [[10,-0.65, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.65, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 20, -0.5 , 0.03,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.40, 0.1,  30,   2],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [10, -0.7, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20],
                [ 2,-0.78, 0.2, 10,  -0.6, 0.2, 10, -0.45, 0.2,  30,  20]]

# einzelner Peak
GrenzeUnten = -0.9
GrenzeOben = -0.3
Position1 = []
Position2 = []
Position1Err = []
Position2Err = []
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
    # Print = f'{popt[0]:.3f} $\pm$ {perr[0]:.3f} & {popt[1]:.3f} $\pm$ {perr[1]:.3f} & {popt[2]:.3f} $\pm$ {perr[2]:.3f} & {popt[3]:.3f} $\pm$ {perr[3]:.3f} & {popt[4]:.3f} $\pm$ {perr[4]:.3f} & {popt[5]:.3f} $\pm$ {perr[5]:.3f} & {popt[6]:.3f} $\pm$ {perr[6]:.3f} & {popt[7]:.3f} $\pm$ {perr[7]:.3f} & {popt[8]:.3f} $\pm$ {perr[8]:.3f} & {popt[9]:.3f} $\pm$ {perr[9]:.3f} & {popt[10]:.3f} $\pm$ {perr[10]:.3f} & {chisq:.3f} & {chi_ndf:.3f} \\\\'
    # Print = Print.replace('.', ',')
    # print(Print)
    plt.plot(xdata_f, g3(xdata_f, *popt), color = 'red',linestyle ='-', label =f'Anpassungskurve', zorder = 10, alpha = 0.7) 
    plt.legend( loc='upper right')
    plt.title(f'Intensität in Abhängigkeit des Winkels für verschiedene Magnetfelder')
    plt.grid()
    plt.ylabel('Intensität $I$ /A')
    plt.xlabel('Winkel $\\alpha$ /°')
    plt.savefig(f'{input_dir}\\Output\\ZM\\Peak_{f[i][-5]}A.pdf')
    plt.cla()
    Position1.append(popt[4]-popt[1])
    Position1Err.append(np.sqrt(perr[4]**2 + perr[1]**2))
    Position2.append(-popt[4]+popt[7])
    Position2Err.append(np.sqrt(perr[4]**2 + perr[7]**2))
    
    # Position1.append(f'{(popt[4] - popt[1]):.5f} $\pm$ {(np.sqrt(perr[4]**2 + perr[1]**2)):.5f} ')
    # Position2.append(f'{(popt[7] - popt[4]):.5f} $\pm$ {(np.sqrt(perr[7]**2 + perr[4]**2)):.5f} ')
BFeld = np.array([0, 58, 129, 202, 272, 333, 382, 421, 450, 475])
xErr = []
for i in BFeld:
    xErr.append(i*0.025)

plt.errorbar(BFeld, Position1, xerr=xErr, yerr=Position1Err,    color = 'red',linestyle ='-', label =f'$|\mu_2 - \mu 1|$', zorder = 10, alpha = 0.7) 
plt.errorbar(BFeld, Position2, xerr=xErr, yerr=Position2Err,    color = 'blue',linestyle ='-', label =f'$|\mu_2 - \mu 3|$', zorder = 10, alpha = 0.7)
popt1, pcov1 = curve_fit(lin, BFeld, Position1, p0=[1, 1], sigma=np.asarray(Position1Err), absolute_sigma=True, maxfev=10000)
popt2, pcov2 = curve_fit(lin, BFeld, Position2, p0=[1, 1], sigma=np.asarray(Position2Err), absolute_sigma=True, maxfev=10000)
perr1 = np.sqrt(np.diag(pcov1))
perr2 = np.sqrt(np.diag(pcov2))
ydatapre1 = lin(BFeld, *popt1)
ydatapre2 = lin(BFeld, *popt2)
chisq1 = np.sum(((Position1-ydatapre1)**2)/np.asarray(Position1Err)**2)
chisq2 = np.sum(((Position2-ydatapre2)**2)/np.asarray(Position2Err)**2)
chi_ndf1 = chisq1/(len(BFeld)-2)
chi_ndf2 = chisq2/(len(BFeld)-2)
Print1 = f'{popt1[0]} $\pm$ {perr1[0]} & {popt1[1]} $\pm$ {perr1[1]} & {chisq1} & {chi_ndf1} \\\\'
Print2 = f'{popt2[0]} $\pm$ {perr2[0]} & {popt2[1]} $\pm$ {perr2[1]} & {chisq2} & {chi_ndf2} \\\\'
Print1 = Print1.replace('.', ',')
Print2 = Print2.replace('.', ',')
print(Print1)
print(Print2)
plt.plot(BFeld, lin(BFeld, *popt1), color = 'red',
        label =f'$|\mu_2 - \mu 1|$ Anpassung', zorder = 10)
plt.plot(BFeld, lin(BFeld, *popt2), color = 'blue',
        label =f'$|\mu_2 - \mu 3|$ Anpassung', zorder = 10)
plt.legend()
plt.title(f'Aufspaltung der Maxima in Abhängigkeit des Magnetfeldes')
plt.grid()
plt.ylabel('Aufspaltung $\\Delta \\alpha$ /°')
plt.xlabel('Magnetfeld $B$ / mT')
plt.savefig(f'{input_dir}\\Output\\ZM\\AufspaltungZuBFeld.pdf')
plt.cla()





# Nochmal aber mit Filter

BFeld1 = np.array([333, 382, 421, 450, 475])
BFeld2 = np.array([272, 333, 382, 421, 450, 475])
Position1 = Position1[5:]
Position2 = Position2[4:]
Position1Err = Position1Err[5:]
Position2Err = Position2Err[4:]
plt.errorbar(BFeld1, Position1, yerr=Position1Err,    color = 'red',linestyle ='-', label =f'$|\mu_2 - \mu 1|$', zorder = 10, alpha = 0.7) 
plt.errorbar(BFeld2, Position2, yerr=Position2Err,    color = 'blue',linestyle ='-', label =f'$|\mu_2 - \mu 3|$', zorder = 10, alpha = 0.7)
popt1, pcov1 = curve_fit(lin, BFeld1, Position1, p0=[1, 1], sigma=np.asarray(Position1Err), absolute_sigma=True, maxfev=10000)
popt2, pcov2 = curve_fit(lin, BFeld2, Position2, p0=[1, 1], sigma=np.asarray(Position2Err), absolute_sigma=True, maxfev=10000)
perr1 = np.sqrt(np.diag(pcov1))
perr2 = np.sqrt(np.diag(pcov2))
ydatapre1 = lin(BFeld1, *popt1)
ydatapre2 = lin(BFeld2, *popt2)
chisq1 = np.sum(((Position1-ydatapre1)**2)/np.asarray(Position1Err)**2)
chisq2 = np.sum(((Position2-ydatapre2)**2)/np.asarray(Position2Err)**2)
chi_ndf1 = chisq1/(len(BFeld1)-2)
chi_ndf2 = chisq2/(len(BFeld2)-2)
Print1 = f'{popt1[0]} $\pm$ {perr1[0]} & {popt1[1]} $\pm$ {perr1[1]} & {chisq1} & {chi_ndf1} \\\\'
Print2 = f'{popt2[0]} $\pm$ {perr2[0]} & {popt2[1]} $\pm$ {perr2[1]} & {chisq2} & {chi_ndf2} \\\\'
Print1 = Print1.replace('.', ',')
Print2 = Print2.replace('.', ',')
print(Print1)
print(Print2)
plt.plot(BFeld1, lin(BFeld1, *popt1), color = 'red',
        label =f'$|\mu_2 - \mu 1|$ Anpassung', zorder = 10)
plt.plot(BFeld2, lin(BFeld2, *popt2), color = 'blue',
        label =f'$|\mu_2 - \mu 3|$ Anpassung', zorder = 10)
plt.legend()
plt.title(f'Aufspaltung der Maxima in Abhängigkeit des Magnetfeldes')
plt.grid()
plt.ylabel('Aufspaltung $\\Delta \\alpha$ /°')
plt.xlabel('Magnetfeld $B$ / mT')
plt.savefig(f'{input_dir}\\Output\\ZM\\AufspaltungZuBFeld_Filter.pdf')
plt.cla()


    




# for i in range(len(Position1)):
#     Position1[i] = Position1[i].replace('.', ',')
#     Position2[i] = Position2[i].replace('.', ',')
#     print(f'{i} & {Position1[i]} & {Position2[i]} \\\\')
# # Peaks eines Spektrums
# # Intensitäten einzeln

# data = read_csv_input(f[0], 0, 0)
# xdata = np.asarray(data['a'], dtype=np.float64)
# ydata = np.asarray(data['I1'], dtype=np.float64)
# yerr = np.asarray(0.02*abs(data['I1']+0.01), dtype=np.float64)

# xk = [5.5 ]
# yk = []
# for i in range(2, len(ydata)):
#     if ydata[i] < ydata[i-1] and ydata[i-1] >= ydata[i-2] and ydata[i]> 7:
#         if xk[-1] > xdata[i-1]-0.01:   
#             xk.append(xdata[i-1])
#             yk.append(ydata[i-1])
# plt.scatter(xk[1:], yk, marker='x', color='red')
# plt.errorbar(xdata, ydata, yerr=yerr, color = 'black')
# plt.title(f'Intensität in Abhängigkeit des Winkels ohne Magnetfeld')
# plt.grid()
# plt.ylabel('Intensität $I$ /A')
# plt.xlabel('Winkel $\\alpha$ /°')
# plt.savefig(f'{input_dir}\\Output\\ZM\\Peakabstände.pdf')
# plt.cla()

# xValues = []
# Abstand = []
# for i in range(1, len(xk)):
#     Abstand.append(abs(xk[i]-xk[i-1]))
#     xValues.append(0.5*(xk[i]+xk[i-1]))
# plt.scatter(xValues, Abstand, color = 'black', marker='x')
# plt.title(f'Abstände der Maxima')
# plt.grid()
# plt.ylabel('Abstand $\\Delta \\alpha$ /°')
# plt.xlabel('Winkel $\\alpha$ /°')
# plt.savefig(f'{input_dir}\\Output\\ZM\\Abstände.pdf')
# plt.cla()

# i = 0
# while i < len(xValues):
#     if xValues[i] > -3.5 and xValues[i] < 5 or i == 17:
#         i += 1
#     else:
#         xValues.pop(i), Abstand.pop(i), xk.pop(i)
# xk.pop(0)
# plt.scatter(xValues, Abstand, color = 'black', marker='x')
# plt.title(f'Gefilterte Abstände der Maxima')
# plt.grid()
# plt.ylabel('Abstand $\\Delta \\alpha$ /°')
# plt.xlabel('Winkel $\\alpha$ /°')
# plt.savefig(f'{input_dir}\\Output\\ZM\\Abstände_filter.pdf')
# plt.cla()
# Abstand[27] = 0.4


# Finesse = []
# FinesseErr = []
# for i in range(len(xk)-1):
#     p0 = [1/xk[i]**2, xk[i], 1/200, 6]
#     GrenzeUnten = xk[i] - Abstand[i]*0.4
#     GrenzeOben = xk[i] + Abstand[i]*0.4
#     xdata_f = xdata[np.where((xdata > GrenzeUnten) & (xdata < GrenzeOben))]
#     ydata_f = ydata[np.where((xdata > GrenzeUnten) & (xdata < GrenzeOben))]
#     yerr_f = yerr[np.where((xdata > GrenzeUnten) & (xdata < GrenzeOben))]
#     popt, pcov = curve_fit(g1, xdata_f, ydata_f, p0=p0, bounds=([0, GrenzeUnten, 0, 0], [100, GrenzeOben, 1/10, 6]), sigma=yerr_f, absolute_sigma=True, maxfev=1e9)
#     perr = np.sqrt(np.diag(pcov))
#     ydatapre = g1(xdata_f, *popt)
#     chisq = np.sum(((ydata_f-ydatapre)**2)/(abs(ydata_f)*0.01)**2)
#     chi_ndf = chisq/(len(xdata_f)-5)
#     # Print = f'{popt[0]:.4f} $\pm$ {perr[0]:.2g} & {popt[1]:.5f} $\pm$ {perr[1]:.2g} & {popt[2]:.5f} $\pm$ {perr[2]:.2g} & {2*np.sqrt(2*np.log(2))*popt[2]:.3f} $\pm$ {2*np.sqrt(2*np.log(2))*popt[2]:.2g} & {Abstand[i]/(2*np.sqrt(2*np.log(2))*popt[2]):.3f} $\pm$ {Abstand[i]*perr[2]/(2*np.sqrt(2*np.log(2))*popt[2]**2):.3f} & {chi_ndf:.0f} \\\\'
#     # Print = Print.replace('.', ',')
#     # print(Print)
#     Finesse.append(Abstand[i]/(2*np.sqrt(2*np.log(2))*popt[2]))
#     FinesseErr.append(Abstand[i]*perr[2]/(2*np.sqrt(2*np.log(2))*popt[2]**2))
    
#     plt.plot(xdata_f, g1(xdata_f, *popt), color = 'red',linestyle ='-', zorder = 10, alpha = 0.5) 
# plt.errorbar(xdata, ydata, yerr=yerr, color = 'black')
# plt.xlim(-3.9, 5.5)
# plt.title(f'Intensität in Abhängigkeit des Winkels ohne Magnetfeld')
# plt.grid()
# plt.ylabel('Intensität $I$ /A')
# plt.xlabel('Winkel $\\alpha$ /°')
# plt.savefig(f'{input_dir}\\Output\\ZM\\Gausfits.pdf')
# plt.cla()
# print(np.mean(Finesse))
# print(np.sqrt(np.std(Finesse)**2 + np.mean(FinesseErr)**2))
# print(len(Finesse))