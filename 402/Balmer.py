import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import os
import scipy.odr as odr
from sklearn.metrics import r2_score

# --------------------------------------------------------------------
# Vorbereitung
# --------------------------------------------------------------------

# Systempfad
input_dir = os.path.dirname(os.path.abspath(__file__))


# Einlesen der Daten
def read_csv_input(test, head, skip):
    data = pd.read_csv(f'{input_dir}\\{test}', sep="\t", header=head, skiprows=skip).astype(np.float64)
    return data

# Lineare Funktion
def linear(Para, x):
    return Para[0]*x
def lin(x, a):
    return a*x 

# 1 Lorenzfunktionen
def l1(x, h, x0, a, c):
    return  h/(1+((x-x0)*2/a)**2) + c 

# Lorenzfunktionsumme
def l2(x, h1, x1, a1, c1, h2, x2, a2, c2):
    Summe = l1(x, h1, x1, a1, 0) + l1(x, h2, x2, a2, 0) + (c1 + c2)/2
    return Summe



# --------------------------------------------------------------------
# Gitterkonstante
# --------------------------------------------------------------------
data = read_csv_input('Daten\\Balmer\\Gitter.csv', 0, 0)
# Daten einlesen und Fehler definieren
WinkelG = np.asarray(data['Winkel'], dtype=np.float64)
Lambda = np.asarray(data['Lambda'], dtype=np.float64)*1e-9
WinkelGErr = [0.3]*len(WinkelG)
Beta = WinkelG + 150 - 180
BetaErr = [np.sqrt(2)*0.3]*len(WinkelG)

# Sinussumme
Temp = np.sin(np.deg2rad(WinkelG)) + np.sin(np.deg2rad(Beta))
TempErr = np.sqrt((np.deg2rad(WinkelGErr)*np.cos(np.deg2rad(WinkelG)))**2 + (np.deg2rad(BetaErr)*np.cos(np.deg2rad(Beta)))**2)
GitterErr = Lambda*np.sqrt((WinkelGErr*np.cos(np.deg2rad(WinkelG))/np.sin(WinkelG)**2)**2 + (BetaErr*np.cos(np.deg2rad(Beta))/np.sin(Beta)**2)**2)

# Fit
TempValues = np.linspace(min(Temp), max(Temp), 100)
model = odr.Model(linear)
mydata = odr.RealData(Temp, Lambda, sx = TempErr, sy =[1e-12]*len(Lambda))
myodr = odr.ODR(mydata, model, beta0=[4e-7], maxit=10000)
out = myodr.run()
rsquared = r2_score(Lambda, linear(out.beta, Temp))

# Plot
plt.errorbar(Temp, Lambda, xerr=TempErr, yerr=1e-12, marker='.', label='Datenpunkte', linestyle='None', color='blue', capsize=3)
plt.errorbar([Temp[-1]]*2, [690.752e-9, 671.643e-9], xerr=TempErr[-1], yerr=1e-12, marker='.', label='???', linestyle='None', color='green', capsize=3)
plt.plot(TempValues, linear(out.beta, TempValues), label='Anpassungsfunktion', c='red')
plt.title('Gitterkonstante nach $g = \\dfrac{\\lambda}{\sin(\\alpha) + \sin(\\beta)}$', fontsize = 14)
plt.xlabel('$\sin(\\alpha) + \sin(\\beta)$ ', fontsize = 12)
plt.ylabel('$\lambda$ / nm', fontsize = 12)
plt.yticks([400e-9, 450e-9, 500e-9, 550e-9, 600e-9, 650e-9, 700e-9], ['400', '450', '500', '550', '600', '650', '700'])
plt.grid()
plt.legend()
plt.savefig(f'{input_dir}\\Plots\\Balmer\\Gitter.pdf')
plt.show
plt.cla()
print()
print('Gitterkonstante')
print('$Parameter:', out.beta, out.sd_beta,  '$')
print('$R_{lin}^2 =',rsquared, '$')
print(f'Numerisch einzeln: {np.mean(Lambda/Temp), np.std(Lambda/Temp)}')

# Gitterkonstante definieren für spätere Berechnungen
g = out.beta
gErr = out.sd_beta





# --------------------------------------------------------------------
# Balmer-Serie
# --------------------------------------------------------------------

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
dLambda = []
dLambdaErr = []
WinkelListe = [72, 56, 49]
for i in range(len(e)):
    data = read_csv_input(f'Daten\\Balmer\\{e[i]}', 0, 0)
    xValues = np.linspace(Grenze[0][i], Grenze[2][i], 100)
    x = np.asarray(data['a'], dtype=np.float64)
    y = np.asarray(data['I'], dtype=np.float64)

    xdata = x[np.where((x > Grenze[0][i]) & (x < Grenze[2][i]))]
    ydata = y[np.where((x > Grenze[0][i]) & (x < Grenze[2][i]))]
    xErr = [0.001]*len(xdata)
    yErr = [0.1]*len(ydata)
    plt.errorbar(x=xdata, y=ydata, xerr=xErr, yerr=yErr, label='Datenpunkte', color='black', zorder = 0)
    plt.grid()
    plt.legend()
    plt.xlabel('Winkel $\\alpha$ / °', fontsize = 12)
    plt.ylabel('Intensität / b.E.', fontsize = 12)
    plt.title(f'$H_\\{e[i][1:-4]}$ - Messdaten', fontsize = 14)
    plt.savefig(f'{input_dir}\\Plots\\Balmer\\{e[i][:-4]}_Daten.pdf')
    plt.cla()
    popt = []
    perr = []
    print()
    print(f'Balmeranpassungen für {e[i][:-4]}')
    for j in range(2):
        xP = xdata[np.where((xdata > Grenze[j][i]) & (xdata < Grenze[j+1][i]))]
        yP = ydata[np.where((xdata > Grenze[j][i]) & (xdata < Grenze[j+1][i]))]
        p, c = curve_fit(l1, xP, yP, p0=[*p0[i][3*j+0:3*j+3],p0[i][6]], maxfev = 100000)
        popt.append([p[0], p[1], p[2], p[3]])
        ydatapre = l1(xP, *popt[j])
        chisq = np.sum(((yP-ydatapre)**2)/ydatapre)
        chi_ndf = chisq/(len(xP)-len([*p0[i][3*j+0:3*j+3],p0[i][6]]))
        perr.append(np.sqrt(np.diag(c)))
        plt.plot(xValues, l1(xValues, *popt[j]), label = 'Anpassung 1', zorder = 10, alpha = 0.8, c = ColorTable[j])
        Print = (f'{e[i][:1]}$_\{e[i][1:-4]}$ & {popt[j][0]:.3f} $\pm$ {perr[j][0]:.2g} & {popt[j][1]:.5f} $\pm$ {perr[j][1]:.2g} & {popt[j][2]:.5f} $\pm$ {perr[j][2]:.2g} & {popt[j][3]:.3f} $\pm$ {perr[j][3]:.2g} & {chi_ndf:.4g} \\\\')
        Print = Print.replace('.', ',')
        print(Print)
    plt.xlim(Grenze[0][i], Grenze[2][i])
    plt.errorbar(x=xdata, y=ydata, xerr=xErr, yerr=yErr, label='Datenpunkte', color='green', zorder = 0)
    if i == 2:
        poptTotal, cov = curve_fit(l2, xdata, ydata, p0=[*popt[0], *popt[1]], maxfev = 100000)
        plt.plot(xValues, l2(xValues, *poptTotal), label = 'Summe', zorder = 10, alpha = 0.8, c = 'black')
    plt.grid()
    plt.legend()
    plt.xlabel('Winkel $\\alpha$ / °', fontsize = 12)
    plt.ylabel('Intensität / b.E.', fontsize = 12)
    plt.title(f'$H_\\{e[i][1:-4]}$ - Anpassung', fontsize = 14)
    plt.savefig(f'{input_dir}\\Plots\\Balmer\\{e[i][:-4]}.pdf')
    plt.cla()
    dLambda.append([popt[0][1], popt[1][1]])
    dLambdaErr.append([perr[0][1], perr[1][1]])
    
    # Linienbreite
    print()
    print('Linienbreite für ', e[i][:-4])
    print((g*popt[0][1]*np.cos(np.deg2rad(WinkelListe[i])))[0], np.sqrt((gErr*popt[0][1]*np.cos(np.deg2rad(WinkelListe[i])))**2  + (g*perr[0][1]*np.cos(np.deg2rad(WinkelListe[i])))**2 + (popt[0][1]*g*np.deg2rad(0.5)*np.sin(np.deg2rad(WinkelListe[i])))**2)[0])
    print((g*popt[1][1]*np.cos(np.deg2rad(WinkelListe[i])))[0], np.sqrt((gErr*popt[1][1]*np.cos(np.deg2rad(WinkelListe[i])))**2  + (g*perr[1][1]*np.cos(np.deg2rad(WinkelListe[i])))**2 + (popt[1][1]*g*np.deg2rad(0.5)*np.sin(np.deg2rad(WinkelListe[i])))**2)[0])



# --------------------------------------------------------------------
# Rydberg-Konstante bestimmen
# --------------------------------------------------------------------

x = []
y = []
dy = []
yerr = []
# Wertepaare für die Rydberg-Konstante isolieren
for i in range(3):
    A = - g*dLambda[i][0]*np.cos(np.deg2rad(WinkelListe[i])) + g*dLambda[i][1]*np.cos(np.deg2rad(WinkelListe[i]))
    B = dLambdaErr[i][0]*np.sin(np.deg2rad(WinkelListe[i])) - dLambdaErr[i][1]*np.sin(np.deg2rad(WinkelListe[i]))
    x.append(1/4 - 1/((3+i)**2) )
    y.append(1/(g*(np.sin(np.deg2rad(WinkelListe[i])) + np.sin(np.deg2rad(WinkelListe[i]+150 - 180))))[0])
    dy.append(1/(g*(np.sin(np.deg2rad(WinkelListe[i])) + np.sin(np.deg2rad(WinkelListe[i]-dLambda[i][1]+dLambda[i][0]+150 - 180))))[0])
    yerr.append(np.sqrt((gErr/(g**2 * (np.sin(np.deg2rad(WinkelListe[i])) + np.sin(np.deg2rad(WinkelListe[i]+dLambda[i][1]-30)))))**2 + ((np.deg2rad(0.5)*(np.cos(np.deg2rad(WinkelListe[i])) + np.cos(np.deg2rad(WinkelListe[i] - 30)))) / (g * (np.sin(np.deg2rad(WinkelListe[i])) + np.sin(np.deg2rad(WinkelListe[i] - 30)))**2))**2)[0])

# Fit
popt, cov = curve_fit(lin, x, y, sigma=yerr, absolute_sigma=True)
popt2, cov2 = curve_fit(lin, x, dy, sigma=yerr, absolute_sigma=True)
xValues = np.linspace(min(x), max(x), 100)

# Plot
plt.errorbar(x, y, yerr=yerr, marker='.', linestyle='None', label='Datenpunkte 1', color='blue', linewidth=2,markersize=5)
plt.errorbar(x, dy, yerr=yerr, marker='.', linestyle='None', label='Datenpunkte 2', color='red', linewidth=1,markersize=2)
plt.plot(xValues, lin(xValues, *popt), label='Anpassung 1', c='blue', linewidth=2, markersize=5)
plt.plot(xValues, lin(xValues, *popt2), label='Anpassung 2', c='red', linewidth=1, markersize=1)
plt.plot()
plt.grid()
plt.legend()
plt.xlabel('${1}/{n^2} - {1}/{m^2}$', fontsize = 12)
plt.ylabel('$\\frac{1}{\\lambda}$ / nm$^{-1}$', fontsize = 12)
plt.title('Balmer-Serie', fontsize = 14)
plt.savefig(f'{input_dir}\\Plots\\Balmer\\Balmer.pdf')
plt.show
plt.cla()

# Rydberg-Konstante für H und D samt Unsicherheit
print()
print('Rydberg-Konstante für H und D')
print(popt[0]*1e-7, np.sqrt(np.diag(cov))[0]*1e-7)
print(popt2[0]*1e-7, np.sqrt(np.diag(cov2))[0]*1e-7)






