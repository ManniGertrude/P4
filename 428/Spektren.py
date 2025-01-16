import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.odr as odr
import glob


fig, ax = plt.subplots(layout='constrained')
path = os.path.dirname(os.path.abspath(__file__))


# Verwende glob, um alle .txt Dateien im DatenD1 Ordner zu finden
files = glob.glob(f'{path}\\DatenD2\\*.txt')

for file in files:
    print(file[45:-4])
    data = pd.read_csv(file, sep="\t", header=0, names=['b', 'R'], dtype=str)


    data['b'] = data['b'].str.replace(',', '.').astype(float)
    data['R'] = data['R'].str.replace(',', '.').astype(float)

    xData = data['b'].values
    yData = data['R'].values

    plt.errorbar(xData, yData, xerr=0, yerr=np.sqrt(yData), label=f'{file[45:-4]}', marker='.', zorder=1, alpha = 0.5)
    plt.grid()
    plt.legend()
    plt.savefig(f'{path}\\Plots\\{file[45:-4]}.pdf')
    plt.cla()
    plt.clf()
