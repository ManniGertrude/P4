import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.dirname(os.path.abspath(__file__))
fig, ax = plt.subplots()


epsilon0 = 8.854187817e-12
e = 1.602176634e-19
me = 9.10938356e-31
UG = 30 
dt = 1e-11 
c = ['red', 'black', 'blue', 'green', 'purple', 'cyan', 'orange', 'yellow', 'pink', 'brown']
v_hf = np.sqrt(2 * 7.8507E-19/ me)

for k in range(0, 5000):
    x = [0]  
    v = [0]  
    i = 0
    j = 0
    print(k)
    while x[-1] < 0.05 and i < 100000:
        i += 1
        if x[-1] < 0.05:
            a = e * k / (10*me)
        else:
            a = -e * UG / me
        
        v.append(v[-1] + a * dt)
        x.append(x[-1] + v[-1] * dt)
        if v[-1] > v_hf:
            v[-1] = 0
            j += 1
    if i == 100000:
        ax.scatter(k/10, 0, color=c[j], s=1, marker='o')
    else:
        ax.scatter(k/10, v[-1], color=c[j], s=1, marker='o')
ax.set_xlabel('Beschleunigungsspannung in V')
ax.set_ylabel('Endgeschwindigkeit in m/s')
ax.set_title('Frank-Hertz Simulation')
ax.legend()
plt.grid()
plt.savefig(f'{path}\\FrankHertz.png')
# plt.show()