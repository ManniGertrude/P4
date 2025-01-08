import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
import os

path = os.path.dirname(os.path.abspath(__file__))

def cosinus(Para, x):
    return Para[0] * np.cos(Para[1]-x)**2 + Para[2]

# Polarisator
Untergrund = 0.37
Winkel = np.deg2rad(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]))
WinkelErr = np.deg2rad(np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
Intensität = np.array([2.24, 3.65, 5.45, 7.2, 8.4, 9.5, 10.1, 10.1, 9.33, 8.27, 6.48, 4.62, 3.18, 1.64, 0.77, 0.4, 0.63, 1.42, 2.57]) - Untergrund
IntensitätErr = np.array([0.03, 0.05, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.05, 0.03, 0.05, 0.05])

model = odr.Model(cosinus)
data = odr.RealData(Winkel, Intensität, sx=WinkelErr, sy=IntensitätErr)
odr = odr.ODR(data, model, beta0=[1, 1, 1])
out = odr.run()

xVals = np.linspace(0, np.pi, 100)
yVals = cosinus(out.beta, xVals,)

residuals = Intensität - cosinus(out.beta, Winkel)
chisq = np.sum((residuals / IntensitätErr) ** 2)
chisq_red = chisq / (len(Winkel) - len(out.beta))
print(chisq_red)
ax, fig = plt.subplots()
plt.errorbar(Winkel, Intensität, yerr=IntensitätErr, xerr=WinkelErr, fmt='.', label='Messwerte', capsize=2, color='black')
plt.plot(xVals, yVals, label='Anpassung', color='red')
plt.xlabel('Winkel [rad]')
plt.ylabel('Photodiodenspannung [mV]')
plt.title('Laserintensiät in Abhängigkeit des Polarisatorwinkels')
plt.grid()
plt.legend()
print(out.beta, out.sd_beta)
plt.savefig(f'{path}\\Polarisator.pdf')
plt.show