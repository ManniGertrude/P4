import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.optimize import curve_fit
path = os.path.dirname(os.path.abspath(__file__))



# Lineares Modell
def linear(x, a, b):
    return a * x + b



LambdaListe = np.array([546, 365, 365, 365, 365, 405, 405, 436, 436, 578, 578, 546])


NamenListe = ['546_1', '365_1', '365_2', '365_3', '365_4', '405_1', '405_2', '436_1', '436_2', '578_1', '578_2', '546_2']


U_G_list = [[0.492,0.391,0.297,0.248,0.18,0.125,0.078,0.026,0.],
               [1.74,1.54,1.17,0.78,0.66,0.515,0.41,0.255,0.115,0.066,0],
               [1.608,1.357,1.046,0.948,0.816,0.686,0.548,0.431,0.303,0.212,0.104,0],
               [1.76,1.434,1,0.84,0.718,0.633,0.47,0.322,0.184,0.077,0],
               [1.2,1.11,0.967,0.885,0.802,0.688,0.576,0.406,0.307,0.208,0.104,0], 
               [1.192,1.077,0.901,0.7,0.6,0.493,0.395,0.301,0.202,0.105,0],
               [1.02,0.895,0.702,0.605,0.499,0.406,0.302,0.21,0.099,0.04,0],
               [1.092,0.994,0.816,0.687,0.604,0.506,0.390,0.299,0.195,0.104,0],
               [1.2,1.104,1.0,0.895,0.796,0.691,0.600,0.504,0.396,0.298,0.186,0],
               [0.4,0.327,0.276,0.22,0.171,0.118,0.073,0.022,0],
               [0.38,0.33,0.28,0.228,0.179,0.128,0.076,0.02,0],
               [0.541,0.496,0.447,0.405,0.3,0.2,0.141,0.096,0.055,0]]


I_A_list = [[0.001,0.006,0.015,0.023,0.03,0.047,0.061,0.08,0.098],
               [0.008,0.133,0.52,1.74,2.45,3.41,4.22,5.63,6.8,7.3,8.0],
               [0.017,0.056,0.195,0.285,0.490,0.795,1.19,1.57,2.06,2.42,2.96,3.36],
               [0.005,0.21,0.73,1.29,1.85,2.4,3.53,4.72,6.1,6.9,7.5],
               [0.076,0.104,0.165,0.227,0.314,0.485,0.714,1.16,1.47,1.84,2.21,2.59],
               [0.007,0.015,0.043,0.088,0.141,0.202,0.276,0.355,0.475,0.555,0.78],
               [0.023,0.042,0.081,0.112,0.151,0.194,0.260,0.338,0.45,0.514,0.57],
               [0.002,0.007,0.033,0.066,0.093,0.133,0.192,0.251,0.333,0.42,0.53],
               [0.006,0.011,0.011,0.02,0.037,0.062,0.091,0.123,0.178,0.233,0.313,0.485],
               [0.00001,0.008,0.016,0.029,0.047,0.074,0.107,0.155,0.19],
               [0.001,0.011,0.018,0.03,0.05,0.076,0.12,0.179,0.207],
               [0.001,0.006,0.006,0.002,0.004,0.008,0.009,0.015,0.023,0.025]]


IndexList = [[1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9, 10],
            [ 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [3, 4, 5, 6, 7, 8, 9],
            [2, 3, 4, 5, 6, 7],
            [ 2, 3, 4, 5, 6, 7, 8],
            [ 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [ 3, 4, 5, 6, 7]]


U0List = []
U0ListErr = []
for i in range(len(LambdaListe)):
    U_G = np.array(U_G_list[i])
    I_A = np.array(I_A_list[i])

    # Fehlerberechnung
    U_G_error = 0.03 * U_G  # 3% Fehler auf U_G
    I_A_error = 0.1 * I_A  # 10% Fehler auf I_A

    # Wurzel des Anodenstroms
    sqrt_I_A = np.sqrt(I_A)
    sqrt_I_A_error = 0.5 * I_A_error / sqrt_I_A  # Fehlerfortpflanzung für sqrt(I_A)

    # Indizes der Punkte, die für den Fit verwendet werden
    fit_indices = IndexList[i]  # Anpassen der Indizes für gewünschte Punkte
    fit_U_G = U_G[fit_indices]
    fit_sqrt_I_A = sqrt_I_A[fit_indices]
    fit_U_G_error = U_G_error[fit_indices]
    fit_sqrt_I_A_error = sqrt_I_A_error[fit_indices]

    # Lineare Regression (mit Fehlergewichtung)
    weights = 1 / fit_sqrt_I_A_error**2  # Gewichtung nach Fehlern
    fit_params, cov_matrix = np.polyfit(fit_U_G, fit_sqrt_I_A, 1, w=weights, cov=True)
    fit_line = np.poly1d(fit_params)

    # Nullstelle (Grenzspannung U_0) und Fehler
    U_0_Test = -fit_params[1] / fit_params[0]
    U_0_error = np.sqrt((cov_matrix[1, 1] / fit_params[0]**2) +((fit_params[1]**2 * cov_matrix[0, 0]) / fit_params[0]**4) -(2 * fit_params[1] * cov_matrix[0, 1] / fit_params[0]**3))#np.sqrt(np.diag(cov_matrix)[0])
 
    U0List.append(U_0_Test)
    U0ListErr.append(U_0_error)
    
    # Berechnung des reduzierten Chi-Quadrat-Wertes
    residuals = fit_sqrt_I_A - fit_line(fit_U_G)  # Residuen
    chisq = np.sum((residuals / fit_sqrt_I_A_error) ** 2)  # Summe der quadrierten normierten Residuen
    chisq_red = chisq / (len(fit_sqrt_I_A) - len(fit_params))  # Freiheitsgrade: Datenpunkte - Fit-Parameter

    # Plot erstellen
    plt.figure(figsize=(10, 6))

    # Fehlerbalken und Datenpunkte
    plt.errorbar(U_G, sqrt_I_A, xerr=U_G_error, yerr=sqrt_I_A_error, fmt='o', label='Messwerte', color='blue', capsize=3, zorder=5)

    # Markierung der Fit-Punkte
    plt.scatter(fit_U_G, fit_sqrt_I_A, color='purple', label='Punkte für Fit', zorder=6)

    # Fit-Linie plotten
    x_fit = np.linspace(0, np.max(U_G) + 0.1, 100)
    plt.plot(x_fit, fit_line(x_fit), label=f'Fit: $y = {fit_params[0]:.3f}x + {fit_params[1]:.3f}$', color='lightgreen', zorder=4)
    # Nullstelle markieren
    plt.axvline(U_0_Test, color='orange', linestyle='--', label=f'Grenzspannung $U_0 = {U_0_Test:.3f} \pm {U_0_error:.3f}$ V')

    # Plot-Details
    plt.title(f'Anodenstrom bei $\lambda$={LambdaListe[i]} [nm] ({NamenListe[i][-1:]}. Messung) ', fontsize=14)
    plt.xlabel('Gegenspannung $U_G$ [V]', fontsize=12)
    plt.ylabel('$\sqrt{I_A}$ [$\sqrt{nA}$]', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    # plt.tight_layout()

    # Diagramm speichern
    plt.savefig(f'{path}\\Plots\\PWQ\\{NamenListe[i]}.png')

    # Diagramm anzeigen
    plt.show

    # Ergebnisse ausgeben
    # print(f"Die Grenzspannung U_0 beträgt: {U_0:.3f} ± {U_0_error:.3f} V")
    # print(f"Chi-Quadrat-Wert: {chisq:.3f}")
    # print(f"Reduziertes Chi-Quadrat: {chisq_red:.3f}")
    


# Daten: Wellenlängen (in nm), Grenzspannung (in V), und Fehler auf U_0
wavelengths = np.array([365, 405, 436, 546, 578])

U0 = np.array([(U0List[1]+U0List[2]+U0List[3]+U0List[4])/4, (U0List[5]+U0List[6])/2, (U0List[7]+U0List[8])/2, (U0List[11]+U0List[0])/2, (U0List[9]+U0List[10])/2])
U0_errors = np.array([(U0ListErr[1]+U0ListErr[3]+U0ListErr[2]+U0ListErr[4])/2, (U0ListErr[5]+U0ListErr[6])/2, (U0ListErr[7]+U0ListErr[8])/2, (U0ListErr[11]+U0ListErr[0])/2, (U0ListErr[9]+U0ListErr[10])/2])

# Konstanten
c = 3e8  # Lichtgeschwindigkeit in m/s
e = 1.602e-19  # Elementarladung in C

# Umrechnung der Wellenlängen in Frequenzen
frequencies = c / (wavelengths * 1e-9)  # in Hz
frequencies *= 1e-14  # in 10^14 Hz (für bessere Plot-Skalierung)

# Lineare Regression mit Fehlerberücksichtigung
popt, pcov = curve_fit(linear, frequencies, U0, sigma=U0_errors, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))  # Fehler der Fit-Parameter

# Berechnung der Planckschen Konstante und der Austrittsarbeit
h_over_e = popt[0]  # h/e aus der Steigung
h_over_e_error = perr[0]
W_A_over_e = -popt[1]  # -b aus dem Achsenabschnitt
W_A_over_e_error = perr[1]

h = h_over_e * e  # h = (h/e) * e
h_error = h_over_e_error * e
W_A = W_A_over_e * e  # W_A = -(b/e) * e
W_A_error = W_A_over_e_error * e

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(frequencies, U0, yerr=U0_errors, fmt='o', label='Messdaten', color='blue', capsize=3)
plt.plot(frequencies, linear(frequencies, *popt), label=f'Fit: $y = {popt[0]:.3f}x + {popt[1]:.3f}$', color='orange')
plt.xlabel(r'Frequenz $\nu \, / \, 10^{14} \, \mathrm{Hz}$', fontsize=12)
plt.ylabel(r'Grenzspannung $U_0 \, / \, \mathrm{V}$', fontsize=12)
plt.title('Grenzspannung gegen Frequenz', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Speichern des Plots
plt.savefig(f'{path}\\Plots\\PWQ\\FinalerPlot.pdf')
plt.show

# Ergebnisse ausgeben
print("Fit-Ergebnisse:")
print(f"Steigung (h/e): {h_over_e:.3e} ± {h_over_e_error:.3e} V·s")
print(f"Achsenabschnitt (-W_A/e): {W_A_over_e:.3e} ± {W_A_over_e_error:.3e} V")
print()
print("Berechnete Werte:")
print(f"Plancksche Konstante h: {h:.3e} ± {h_error:.3e} J·s")
print(f"Austrittsarbeit W_A: {W_A:.3e} ± {W_A_error:.3e} J")
