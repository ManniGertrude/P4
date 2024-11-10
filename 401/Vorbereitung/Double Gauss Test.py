import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
import pandas as pd
import os
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Pfad und Daten einlesen
path = os.path.dirname(os.path.abspath(__file__))
Data = pd.read_csv(f'{path}\\Data.txt', delimiter=';', skiprows=1, names=['x', 'y'])
fig, ax = plt.subplots()


# Funktionen
# Gauss Funktion
def g1(x, a1, x1, o1, c, m):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-x1)**2/(2*o1)**2)
    return g1 + c + m*x

# Double Gauss Funktion
def g2(x, a1, x1, o1, a2, x2, o2, c, m):
    g1= a1/(o1*(2*np.pi)**(1/2))*np.exp(-(x-x1)**2/(2*o1)**2)
    g2 = a2/(o2*(2*np.pi)**(1/2))*np.exp(-(x-x2)**2/(2*o2)**2)
    return g1 + g2 + c + m*x


# Grenzen für die Daten
G1 = 200
G2 = 360

# Fit
xData = Data['x'][G1:G2]
yData = Data['y'][G1:G2]

# Parameter bounds a1, x1, o1, a2, x2, o2, c, m
p0           = [1, -2, 0.01, 1, -1.85, 0.01, 4, 0.5]
lower_bounds = [0.04, -2.05, 0.001, 0.04, -1.9, 0.01, 3, 0]
upper_bounds = [1.1, -1.95, 0.2, 1.1, -1.8, 0.2, 5, 1]

# Curve Fit
popt, pcov = curve_fit(g2, xData, yData, p0=p0, bounds=(lower_bounds, upper_bounds))
perr = np.sqrt(np.diag(pcov))
rsquaredfil = r2_score(yData,g2(xData, *popt))
ax.plot(Data['x'], Data['y'])
plt.plot(xData, g2(xData, *popt), color = 'red',linestyle ='-', label =f'Anpassungskurve', alpha = 0.7) 
ax.legend()
print(popt)
ax.set_xlim(-2.4, -1.5)
ax.set_ylim(3.9, 5.3)
ax.set_title('Double Gauss Fit')
ax.set_xlabel('Winkel in rad')
ax.set_ylabel('Intensität in a.u.')
ax.grid()
plt.savefig(f'{path}\\Test.png')
# plt.show()