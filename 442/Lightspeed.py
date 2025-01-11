import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
import os

path = os.path.dirname(os.path.abspath(__file__))


def Printer(Temp):
    Temp = Temp.replace('.', ',')
    print(Temp)


def lin(Para, x):
    return Para * x 

x = 1/(2*np.array([0.489, 0.742, 0.985]))
xErr = (2*np.array([0.002, 0.002, 0.002]))
y = np.array([304.056e6, 204.965e6, 150.974e6])
yErr = 0.02e6 * np.ones(3)

model = odr.Model(lin)
data = odr.RealData(x, y, sx=xErr, sy=yErr)
odrVals = odr.ODR(data, model, beta0=[1])
out = odrVals.run()

xVals = np.linspace(min(x), max(x), 100)
yVals = lin(out.beta, xVals,)

chisq = np.sum(((y - lin(out.beta, x))** 2 / yErr**2))
chisq_red = chisq / (len(x) - len(out.beta))
ax, fig = plt.subplots()

Printer(f'{out.beta[0]:.6g} $\pm$ {out.sd_beta[0]:.6g} & {chisq:.2f} & {chisq_red:.2f}')

plt.errorbar(x, y, xerr=xErr, yerr=yErr, fmt='.', label='Messwerte', capsize=2, color='black')
plt.plot(xVals, yVals, label='Anpassung', color='red')
plt.xlabel('(2L)$^{-1}$ [m]', fontsize=12)
plt.ylabel('$\\Delta\\nu_{FSR}$ [Hz]', fontsize=12)
plt.title('Lichtgeschwindigkeitsberechnung', fontsize=14)
plt.grid()
plt.legend()
plt.savefig(f'{path}\\Lightspeed.pdf')
plt.show
