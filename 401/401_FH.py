import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os
import pandas as pd

input_dir = os.path.dirname(os.path.abspath(__file__))


def read_csv_input(test):
    data = pd.read_csv(f'{input_dir}\\{test}', sep="\t", header=0).astype(np.float64)
    return data


def gaus_sep(x, a,o,m, k):
    return a/(o*(2*np.pi)**(1/2))*np.exp(-(x-m)**2/(2*o)**2) + k


def individual_gaus(data, Name):
    xdata = np.asarray(data['U_B'], dtype=np.float64)
    ydata = np.asarray(data['U_A'], dtype=np.float64)
    xerr = np.asarray(0.025*abs(data['U_B'])+0.01)
    yerr = np.asarray(0.025*abs(data['U_A'])+0.01)
    plt.grid()

    GrenzeUnten = [10, 14, 18.5, 23.5, 28.5, 33.5]
    GrenzeOben = [14, 18.2, 23.2, 28.2, 33.4, 38.2]
    P0Values = [[12, 0.5, 11, 0.5], [10, 0.2, 16, 0.5], [10, 0.2, 21, 0.5], [10, 0.2, 26, 0.5], [10, 0.2, 31, 0.5], [10, 0.2, 36, 0.5]]
    for i in range (len(GrenzeUnten)):
        xdata_filtered = xdata[np.where((xdata > GrenzeUnten[i]) & (xdata < GrenzeOben[i]))]
        ydata_filtered = ydata[np.where((xdata > GrenzeUnten[i]) & (xdata < GrenzeOben[i]))]
        xerr_filtered = xerr[np.where((xdata > GrenzeUnten[i]) & (xdata < GrenzeOben[i]))]
        yerr_filtered = yerr[np.where((xdata > GrenzeUnten[i]) & (xdata < GrenzeOben[i]))]
        p0 = P0Values[i]
        popt, pcov = curve_fit(gaus_sep, xdata_filtered, ydata_filtered, p0=p0,
                                              sigma=yerr_filtered, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        ydatapre = gaus_sep(xdata_filtered, *popt)
        chisq = np.sum(((ydata_filtered-ydatapre)**2)/yerr_filtered**2)
        chi_ndf = chisq/(len(xdata_filtered)-len(p0))
        Print = f'{popt[0]:.3f} $\pm$ {perr[0]:.3f} & {popt[2]:.3f} $\pm$ {perr[2]:.3f} & {popt[1]:.3f} $\pm$ {perr[1]:.3f} & {popt[3]:.3f} $\pm$ {perr[3]:.3f} & {chisq:.2f} & {chi_ndf:.2f} \\\\'
        Sigma.append(popt[1])
        SigmaErr.append(perr[1])
        max.append(popt[2])
        may.append(gaus_sep(popt[2], *popt))
        maxerr.append(perr[2]+ 0.05*may[-1])
        mayerr.append(gaus_sep(popt[2], *perr))
        Print = Print.replace('.', ',')
        # print(Print)
        plt.plot(xdata_filtered, gaus_sep(xdata_filtered, *popt), color = ColorTable[i],
                label=f'Gaus {i+1} mit $\chi^2/ndf$ = {chi_ndf:.2f} ', zorder = 10)
    plt.errorbar(xdata, ydata, xerr=xerr, yerr= yerr, color='black', marker='o',markersize =1, 
                     linestyle='none', label='Datenpunkte', zorder = 1, alpha = 0.8)
    # print(max)
    bottom, top = plt.ylim()
    plt.ylim(bottom, top*1.2)
    if Name[0] == 'U':
        plt.title(f'Anodenstrom bei $U_G={Name[-3:]}$V, $T=165$°C')
    elif Name[0] == 'T':
        plt.title(f'Anodenstrom bei $U_G=2,7$V, $T={Name[-3:]}$°C')
    plt.xlim(-0.5, 40.5)
    plt.xlabel('Beschleunigungsspannung $U_{B}$ /V')
    plt.ylabel('Anodenstrom $I_{A}$ /A')
    plt.legend()
    plt.grid()
    plt.show
    plt.savefig(f'{input_dir}\\Output\\FH\\{Name}.pdf')
    plt.cla()
    
ax, fig = plt.subplots()
f = ["Data\\Tconst_165\\U2_2.0.csv", 
     "Data\\Tconst_165\\U2_2.7.csv", 
     "Data\\Tconst_165\\U2_3.4.csv", 
     "Data\\Tconst_165\\U2_4.0.csv",
     "Data\\Uconst_2.7\\U2_2_7V_t_165.csv", 
     "Data\\Uconst_2.7\\T_170.csv", 
     "Data\\Uconst_2.7\\T_175.csv", 
     "Data\\Uconst_2.7\\T_180.csv"]

ColorTable = ['crimson','darkgoldenrod','olive', 'green', 'darkcyan',
              'slateblue', 'orchid', 'deeppink', 'purple', 'black']
NameList = ["U2_2,0", "U2_2,7", "U2_3,4", "U2_4,0","T_165", "T_170", "T_175", "T_180"]
Namen = ['U = 2,0V', 'U = 2,7V', 'U = 3,4V', 'U = 4,0V', 'T = 165°C', 'T = 170°C', 'T = 175°C', 'T = 180°C']
max = []
may = []
maxerr = []
mayerr = []
Sigma = []
SigmaErr = []


# Alle Messungen einzeln gegaußt
for i in range(len(f)):
    # print()
    # print(NameList[i])
    # print('Integral $A$ & Erwartungswert $\mu$ & Sigma $\sigma$ & Konstante k $\chi^2$ & $\chi^2$/ndf\\\\ \hline')
    data = read_csv_input(f[i])
    individual_gaus(data, NameList[i])

# Maxima der Gaußfunktionen
for i in range(8):
    if i < 4:
        plt.errorbar(max[6*i:6*i+6], may[6*i:6*i+6], xerr=maxerr[6*i:6*i+6], yerr=mayerr[6*i:6*i+6],
                    marker = '.', label=f'Maxima für $U_B =$ {NameList[i][-3:]} V', zorder = 20, color = ColorTable[i])
    else:
        plt.errorbar(max[6*i:6*i+6], may[6*i:6*i+6], xerr=maxerr[6*i:6*i+6], yerr=mayerr[6*i:6*i+6],
                    marker = '.', label=f'Maxima für $T =$ {NameList[i][-3:]} °C', zorder = 20, color = ColorTable[i])

plt.xlabel('Beschleunigungsspannung $U_{B}$ /V')
plt.ylabel('Anodenstrom $I_{A}$ /A')
plt.legend()
plt.grid()
plt.title('Maxima der Gaußfunktionen')
plt.show
plt.savefig(f'{input_dir}\\Output\\FH\\Maxima.pdf')
plt.cla()

# Normierte Maxima
for i in range(8):
    diff = []
    diffErr = []
    FWHM = []
    FWHMErr = []
    for j in range(6):
        FWHM.append(2*np.sqrt(2*np.log(2))*Sigma[6*i+j])
        FWHMErr.append((2*np.sqrt(2*np.log(2))*SigmaErr[6*i+j] + np.sqrt((FWHM - 2*np.sqrt(2*np.log(2))*np.mean(Sigma[6*i:6*i+6]))**2/6))[0])

        if j == 0:
            continue
        if j == 1:
            continue
        else:
            diff.append(max[6*i+j] - max[6*i+j-1])
            diffErr.append(np.sqrt(maxerr[6*i+j]**2 + maxerr[6*i+j-1]**2))
            plt.errorbar(i + j*0.1, diff[-1], yerr=diffErr[-1], capsize=2, marker = 'o', color = ColorTable[i])
    Sum = np.mean(diff)
    SumErr = np.sqrt(np.sum((diff-Sum)**2)/(len(diff)-1))
    FWHM = np.mean(FWHM)
    FWHMErr = np.mean(FWHMErr)
    Print = f'{Namen[i]} & {Sum:.2f} $\pm$ {SumErr:.2f} & {diff[0]:.2f} $\pm$ {diffErr[0]:.2f} & {diff[1]:.2f} $\pm$ {diffErr[1]:.2f} & {diff[2]:.2f} $\pm$ {diffErr[2]:.2f} & {diff[3]:.2f} $\pm$ {diffErr[3]:.2f}  \\\\'  # & {diff[4]:.2f} $\pm$ {diffErr[4]:.2f}
    Print = Print.replace('.', ',')
    print(Print)
    plt.errorbar(i, Sum, yerr=SumErr, capsize=2, marker = 'o', color = 'black')
plt.xticks([0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25, 7.25], Namen, rotation = 20)
plt.xlabel('Messreihe')
plt.ylabel('Differenz der Maxima / V')
plt.grid()
plt.title('Normierte Maxima der Gaußfunktionen')
plt.show
plt.savefig(f'{input_dir}\\Output\\FH\\NormMaxima.pdf')
plt.cla()



# Alle U-Variationen bei T=165°C
for i in range(4):
    data = read_csv_input(f[i])
    xdata = np.asarray(data['U_B'], dtype=np.float64)
    ydata = np.asarray(data['U_A'], dtype=np.float64)
    xerr = np.asarray(0.025*abs(data['U_B'])+0.01)
    yerr = np.asarray([0.025*abs(data['U_A'])+0.01])
    plt.errorbar(xdata, ydata, xerr = xerr, yerr= yerr, marker='o',markersize =1, color=ColorTable[i], label=f'Daten bei $U_B$ = {NameList[i][-3:]}°C')
plt.xlim(-0.5, 40.5)
plt.grid()
plt.title(f'Variation der Gegenspannung bei $T=165$°C')
plt.xlabel('Beschleunigungsspannung $U_{B}$ /V')
plt.ylabel('Anodenstrom $I_{A}$ /A')
plt.legend()
plt.savefig(f'{input_dir}\\Output\\FH\\U_Var.pdf')
plt.cla()


# Alle T-Variationen bei U=2.7V
for i in range(4, 8):
    data = read_csv_input(f[i])
    xdata = np.asarray(data['U_B'], dtype=np.float64)
    ydata = np.asarray(data['U_A'], dtype=np.float64)
    yerr = np.asarray([0.05*abs(data['U_A'])+0.01])
    plt.errorbar(xdata, ydata, yerr= yerr, marker='o',markersize =1, color=ColorTable[i-4], label=f'Daten bei T = {NameList[i][-3:]}°C')
plt.xlim(-0.5, 40.5)
plt.grid()
plt.title(f'Variation der Temperatur bei $U_G$ = 2,7 V')
plt.xlabel('Beschleunigungsspannung $U_{B}$ /V')
plt.ylabel('Anodenstrom $I_{A}$ /A')
plt.legend()
plt.savefig(f'{input_dir}\\Output\\FH\\T_Var.pdf')
plt.cla()