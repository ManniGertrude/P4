import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
import os

path = os.path.dirname(os.path.abspath(__file__))


def Printer(Temp):
    Temp = Temp.replace('.', ',')
    print(Temp)


def cosinus(Para, x):
    return Para[0] * np.cos(Para[1]-np.deg2rad(x))**2 + Para[2]


# Polarisator
Untergrund = 0 # .37
Winkel = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
WinkelErr = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
Intensität = np.array([2.24, 3.65, 5.45, 7.2, 8.4, 9.5, 10.1, 10.1, 9.33, 8.27, 6.48, 4.62, 3.18, 1.64, 0.77, 0.4, 0.63, 1.42, 2.57])
IntensitätErr = np.array([0.03, 0.05, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.05, 0.03, 0.05, 0.05]) + 0.06

model = odr.Model(cosinus)
data = odr.RealData(Winkel, Intensität, sx=WinkelErr, sy=IntensitätErr)
odrVals = odr.ODR(data, model, beta0=[1, 1, 1])
out = odrVals.run()

xVals = np.linspace(0, 180, 100)
yVals = cosinus(out.beta, xVals,)

chisq = np.sum(((Intensität - cosinus(out.beta, Winkel))** 2 / IntensitätErr**2))
chisq_red = chisq / (len(Winkel) - len(out.beta))
ax, fig = plt.subplots()

Printer(f'{out.beta[0]:.3f} $\pm$ {out.sd_beta[0]:.3f} & {np.rad2deg(out.beta[1]):.2f} $\pm$ {np.rad2deg(out.sd_beta[1]):.2f} & {out.beta[2]:.3f} $\pm$ {out.sd_beta[2]:.3f} & {chisq:.2f} & {chisq_red:.2f}')


plt.errorbar(Winkel, Intensität, yerr=IntensitätErr, xerr=WinkelErr, fmt='.', label='Messwerte', capsize=2, color='black')
plt.plot(xVals, yVals, label='Anpassung', color='red')
plt.xlabel('Polarisatorwinkel [°]', fontsize=12)
plt.ylabel('Photodiodenspannung [mV]', fontsize=12)
plt.title('Photodiodenspannung in Abh. des Polarisatorwinkels', fontsize=14)
plt.grid()
plt.legend()
plt.savefig(f'{path}\\Polarisator.pdf')
plt.show

for i in range(len(Winkel)):
    Printer(f'{Winkel[i]} $\pm$ {WinkelErr[i]} & {Intensität[i]+Untergrund:.2f} $\pm$ {IntensitätErr[i]:.2f} \\\\')