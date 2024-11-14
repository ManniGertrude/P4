import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os
import pandas as pd

input_dir = os.path.dirname(os.path.abspath(__file__))


def read_csv_input(test):
    data = pd.read_csv(f'{input_dir}\\{test}', sep="\t", header=0).astype(np.float64)
    return data


def gaus_sep(x, a,o,m):
    return a/(o*(2*np.pi)**(1/2))*np.exp(-(x-m)**2/(2*o)**2)


def individual_gaus(data, Name):
    xdata = np.asarray(data['U_B'], dtype=np.float64)
    ydata = np.asarray(data['U_A'], dtype=np.float64)
    yerr = np.asarray(0.05*abs(data['U_A'])+0.01)
    plt.grid()
    
    GrenzeUnten = [10, 14, 18.5, 23.5, 28.5, 33.5]
    GrenzeOben = [14, 18.2, 23.2, 28.2, 33.4, 38.2]
    P0Values = [[10, 0.2, 11], [10, 0.2, 16], [10, 0.2, 21], [10, 0.2, 26], [10, 0.2, 31], [10, 0.2, 36]]
    for i in range (len(GrenzeUnten)):
        xdata_filtered = xdata[np.where((xdata > GrenzeUnten[i]) & (xdata < GrenzeOben[i]))]
        ydata_filtered = ydata[np.where((xdata > GrenzeUnten[i]) & (xdata < GrenzeOben[i]))]
        yerr_filtered = yerr[np.where((xdata > GrenzeUnten[i]) & (xdata < GrenzeOben[i]))]
        p0 = P0Values[i]
        popt, pcov = curve_fit(gaus_sep, xdata_filtered, ydata_filtered, p0=p0,
                                              sigma=yerr_filtered, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        ydatapre = gaus_sep(xdata_filtered, *popt)
        chisq = np.sum(((ydata_filtered-ydatapre)**2)/yerr_filtered**2)
        chi_ndf = chisq/(len(xdata_filtered)-len(p0))
        Print = f'{popt[0]:.3f} $\pm$ {perr[0]:.3f} & {popt[1]:.3f} $\pm$ {perr[1]:.3f} & {popt[2]:.3f} $\pm$ {perr[2]:.3f} & {chisq:.2f} & {chi_ndf:.2f} \\\\'
        Print = Print.replace('.', ',')
        print(Print)
        plt.plot(xdata_filtered, gaus_sep(xdata_filtered, *popt), color = ColorTable[i],
                label=f'Gaus {i+1} mit $\chi^2/ndf$ = {chi_ndf:.2f} ', zorder = 10)
    plt.errorbar(xdata, ydata, yerr= yerr, color='black', marker='o',markersize =1, 
                     linestyle='none', label='Datenpunkte', zorder = 1, alpha = 0.8)
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

ColorTable = ['firebrick','darkgoldenrod', 'limegreen', 'darkcyan', 'slateblue', 'orchid']
NameList = ["U2_2,0", "U2_2,7", "U2_3,4", "U2_4,0","T_165", "T_170", "T_175", "T_180"]

# Alle Messungen einzeln gegaußt
for i in range(len(f)):
    print()
    print(NameList[i])
    data = read_csv_input(f[i])
    individual_gaus(data, NameList[i])

# Alle U-Variationen bei T=165°C
for i in range(4):
    data = read_csv_input(f[i])
    xdata = np.asarray(data['U_B'], dtype=np.float64)
    ydata = np.asarray(data['U_A'], dtype=np.float64)
    yerr = np.asarray([0.05*abs(data['U_A'])+0.01])
    plt.errorbar(xdata, ydata, yerr= yerr, marker='o',markersize =1, color=ColorTable[i], label=f'Daten bei $U_B$ = {NameList[i][-3:]}°C')
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