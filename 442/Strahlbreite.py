import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
import os

path = os.path.dirname(os.path.abspath(__file__))


def Printer(Temp):
    Temp = Temp.replace('.', ',')
    print(Temp)


def lin(Para, x):
    return Para[0] *np.sqrt( 1+x**2/Para[1]**2)


# Strahlbreite 
xRes1 = np.array([1.9, 45.4])
yRes1 = np.array([0.9, 1])
xRes2 = np.array([2.3, 50.7, 58.3, 68.3])
yRes2 = np.array([0.8, 1.2, 1.25, 1.4])
xRes3 = np.array([1.3, 47.3, 57.3, 65.3, 74.3, 80.3])
yRes3 = np.array([0.6, 1.2, 1.3, 1.45, 1.65, 1.95])

for i in range(3):
    if i == 0:
        x = xRes1
        y = yRes1
    elif i == 1:
        x = xRes2
        y = yRes2
    else:
        x = xRes3
        y = yRes3
    xErr = np.ones(len(x)) * 0.5
    yErr = np.ones(len(y)) * 0.05

    model = odr.Model(lin)
    data = odr.RealData(x, y, sx=xErr, sy=yErr)
    odrVals = odr.ODR(data, model, beta0=[1, 1])
    out = odrVals.run()

    xVals = np.linspace(0, max(x), 100)
    yVals = lin(out.beta, xVals,)

    chisq = np.sum(((y - lin(out.beta, x))** 2 / yErr**2))
    chisq_red = chisq / (len(x) - len(out.beta))
    ax, fig = plt.subplots()

    Printer(f'{out.beta[0]:.3f} $\pm$ {out.sd_beta[0]:.3f} & {out.beta[1]:.3f} $\pm$ {out.sd_beta[1]:.3f} & {chisq:.2f} & {chisq_red:.2f}')

    plt.errorbar(x, y, yerr=yErr, xerr=xErr, fmt='.', label='Messwerte', capsize=2, color='black')
    plt.plot(xVals, yVals, label='Anpassung', color='red')
    plt.xlabel('Position [cm]', fontsize=12)
    plt.ylabel('Strahlbreite [mm]', fontsize=12)
    plt.title('Strahlbreite in Abh. der Position', fontsize=14)
    plt.grid()
    plt.legend()
    plt.savefig(f'{path}\\Strahlbreite{i+1}.pdf')
    plt.show

