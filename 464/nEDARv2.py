import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits import open as fits_open
from scipy.ndimage import rotate as rotate
from tkinter import *
import os


def Calc(light_file, dark_file, cali_file, rotation=0, oben=0, unten=510):
    # ließt die Fits File
    light = fits_open(light_file)[0].data * 1.0
    dark = fits_open(dark_file)[0].data * 1.0
    cali = fits_open(cali_file)[0].data * 1.0

    # Process the images
    bild = light - dark
    bild_r = rotate(bild, rotation, reshape=False)
    cali_r = rotate(cali, rotation, reshape=False)
    
    spe = np.sum(bild_r[oben:unten], axis=0)
    cal = np.sum(cali_r[oben:unten], axis=0)
    
    return spe, cal

# Export Funktion (nicht wirklich notwendig)
def export(spe, cal, output_filen):
    with open(output_file, "w") as f:
        for i in range(len(spe)):
            f.write(f"{i}\t{spe[i]}\t{cal[i]}\n")


# File Paths definieren
path = os.path.dirname(os.path.dirname(__file__)) # Pfad zu dieser Datei
light_file = f'{path}\\Pfad\\zur\\Datei\\light.fit'
dark_file = f'{path}\\Pfad\\zur\\Datei\\dark.fit'
cali_file = f'{path}\\Pfad\\zur\\Datei\\flat.fit'
output_file = f'{path}\\Pfad\\zur\\Datei\\output.txt'

# Parameter definieren
rotation = 0
oben = 0
unten = 510

# Verarbeitung
spe, cal = Calc(light_file, dark_file, cali_file, rotation, oben, unten)



# # Export (nicht wirklich notwendig aber wenn mans mag?)
# export(spe, cal, output_file)


# Plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(spe, label='Spektrum', c='r')
ax1.set_ylabel('Counts', c='r')

ax2.plot(cal, label='Kalibrierung', c='b')
ax2.set_ylabel('Counts', c='b')

plt.legend()
plt.grid()
plt.xlabel('Kanalnummer')
plt.title('Spektrum und Kalibrierung')
plt.show()