import matplotlib.pyplot as plt
import scipy.odr as odr
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
import statistics as st
import os

path = os.path.dirname(os.path.abspath(__file__))
ax, plt = plt.subplots()

G = 6.67430e-11 # Gravitationskonstante
M1 = 2.389*1.988416e30  # Masse 1 in kg
M2 = 2.327*1.988416e30  # Masse 2 in kg
a = 2e7                 # Halbachse
e_s = 0.8               # Exzentrizität = 0.8
e_e = 0.2               # Exzentrizität = 0.2
e_r = 0.0               # Exzentrizität = 0.0
mu = G * (M1 + M2)      # Gravitationsparameter
theta = np.linspace(0, 6 * np.pi, 1000)

# stark eliptischer Orbit
r_s = a * (1 - e_s**2) / (1 + e_s * np.cos(theta))
v_s = np.sqrt(mu * (2 / r_s - 1 / a))

# eliptischer Orbit
r_e = a * (1 - e_e**2) / (1 + e_e * np.cos(theta))
v_e = np.sqrt(mu * (2 / r_e - 1 / a))

# runder Orbit
r_r = a * (1 - e_r**2) / (1 + e_r * np.cos(theta))
v_r = np.sqrt(mu * (2 / r_r - 1 / a))

plt.plot(theta, v_r, label='v bei rundem Orbit $\epsilon = 0.0$')
plt.plot(theta, v_e, label='v bei eliptischem Orbit $\epsilon = 0.2$')
plt.plot(theta, v_s, label='v bei supereliptischem Orbit $\epsilon = 0.8$')
plt.set_xlabel('Umrundungswinkel (radians)')
plt.set_ylabel('Geschwindigkeit (m/s)')
plt.set_title('Umdrehungsgeschwindigkeit eines Doppelsternsystems')
plt.legend(loc='upper right')
plt.grid(True)
ax.savefig(f'{path}\\4.1.png')
print((1-1e-14)**(65e11))