import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.odr as odr

def gauss(Para, x):
    return Para[0] /(Para[2]*np.sqrt(2*np.pi))*np.exp(-0.5*((x - Para[1])/Para[2])**2) + Para[3]

def gausfit(func, x, y, farbe, beta):

    model = odr.Model(func)
    mydata = odr.RealData(x, y, sx=1e-2, sy=np.sqrt(y))
    myodr = odr.ODR(mydata, model, beta0=beta, maxit=1000)
    out = myodr.run()
    
    fy = func(out.beta, x)
    # print('$Parameter:', out.beta, out.sd_beta,  '$')
    chi2 = np.sum((y - fy)**2/fy)
    print(sum((y-fy)**2/fy))
    ndf = len(x) - len(out.beta)
    plt.plot(x, fy, c=farbe,label = f'Gauß-Anpassung mit $\chi^2/ndf = {chi2/ndf:.3g} $', alpha=0.7)
    return out


path = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(f'{path}\\DatenD1\\NaCl281.97.txt', sep="\t", header=0, names=['b', 'R'], dtype=str)

# Replace commas with dots and convert to float
data['b'] = data['b'].str.replace(',', '.').astype(float)
data['R'] = data['R'].str.replace(',', '.').astype(float)

xData = data['b'].values
yData = data['R'].values

xP1 = xData[150:172]
yP1 = yData[150:172]

xP2 = xData[172:195]
yP2 = yData[172:195]

plt.errorbar(xData, yData,xerr=0, yerr=np.sqrt(yData), color='darkolivegreen', label='Messwerte', marker='.', zorder=1, linestyle='None')
gausfit(gauss, xP2, yP2, 'purple', [426, 20, 0.18, 36.5])
gausfit(gauss, xP1, yP1, 'blue', [200, 18.25, 0.3, 36.5])
plt.ylabel('Anzahl der Messergebnisse pro Sekunde pro Winkeleinstellung [1/s]')
plt.xlabel('Winkel [°]')
plt.legend()
plt.grid()
plt.savefig(f'{path}\\NaCl281.97.png')
