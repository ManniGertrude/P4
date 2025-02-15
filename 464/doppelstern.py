import os 
import pandas as pd
import matplotlib.pyplot as plt


path = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots()

for folder in os.listdir(f'{path}\\BAur'):
    print(folder)
    Data = pd.read_csv(f'{path}\\BAur\\{folder}\\output.dat', sep='\t', names=['n', 'x', 'y'])
    ax.plot(Data['n'], Data['y'], label=folder)
    plt.legend()
    plt.grid()
    plt.savefig(f'{path}\\BAur\\{folder}\\output.pdf')
    plt.cla()