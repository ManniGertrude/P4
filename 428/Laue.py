import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__)

# Parameter für Berechnung von z_Q, θ und deren Fehler
L = 15  # Abstand zwischen Kristall und Film (mm)
delta_x = 1.0  # Fehler in x (mm)
delta_y = 1.0  # Fehler in y (mm)
delta_L = 1.0  # Fehler in L (mm)

# Gitterparameter für Netzebenenabstände
a_0 = 0.564  # nm

# Gegebene Werte für (x, y) in mm
xy_data = [
    (-10, 4), (10, 4), (-11, -1), (11, -1), (-10, -4), (10, -4), 
    (-6, -6), (6, -6), (-15, -8), (15, -8), (-9, -9), (9, -9), 
    (-3, -10), (3, -10), (-8, -15), (8, -15), (-15, 8), (15, 8), 
    (-9, 9), (9, 9), (-3, 9), (3, 9), (-8, 14), (8, 14), 
    (0, 11), (0, -11)
]

# Berechnung von z_Q, Δz_Q, θ und Δθ
x_values, y_values, z_q_values, delta_z_q_values, theta_values, delta_theta_values = [], [], [], [], [], []

for x, y in xy_data:
    x_values.append(x)
    y_values.append(y)
    
    # Berechnung von z_Q
    r_squared = x**2 + y**2
    z_q = np.sqrt(r_squared + L**2) - L
    z_q_values.append(z_q)
    
    # Fehlerberechnung für z_Q
    delta_z_q = np.sqrt(
        (x / np.sqrt(r_squared + L**2) * delta_x)**2 +
        (y / np.sqrt(r_squared + L**2) * delta_y)**2 +
        (L / np.sqrt(r_squared + L**2) * delta_L - delta_L)**2
    )
    delta_z_q_values.append(delta_z_q)
    
    # Berechnung von θ
    r = np.sqrt(r_squared)  # Abstand vom Ursprung
    theta = np.arctan2(r, L) * (180 / np.pi)  # Glanzwinkel in Grad
    theta_values.append(theta)
    
    # Fehlerberechnung für θ
    delta_r = np.sqrt((x / r * delta_x)**2 + (y / r * delta_y)**2)
    delta_theta = np.sqrt(
        (1 / (1 + (r / L)**2) * delta_r / L)**2 +
        (-(r / L**2) / (1 + (r / L)**2) * delta_L)**2
    ) * (180 / np.pi)
    delta_theta_values.append(delta_theta)

# Ausgabe der Ergebnisse: x, y, z_Q, Δz_Q, θ, Δθ
print(f"{'x (mm)':>8} {'y (mm)':>8} {'z_Q (mm)':>10} {'Δz_Q (mm)':>12} {'θ (deg)':>10} {'Δθ (deg)':>10}")
for x, y, z_q, delta_z_q, theta, delta_theta in zip(x_values, y_values, z_q_values, delta_z_q_values, theta_values, delta_theta_values):
    print(f"{x:8.1f} {y:8.1f} {z_q:10.3f} {delta_z_q:12.3f} {theta:10.3f} {delta_theta:10.3f}")

# Daten für Netzebenenabstände
hkl_data = {
    "h": [-3, 3, -4, 4, -3, 3, -6, 6, -4, 4, -3, 3, -1, 1, -2, 2, -4, 4, -3, 3, -1, 1, -6, 6, 0, 0],
    "k": [1, 1, 0, 0, -1, -1, -6, -6, -2, -2, -3, -3, -3, -3, -4, -4, 2, 2, 3, 3, 3, 3, 10, 10, 6, -6],
    "l": [1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 4, 4, 2, 2]
}

# Berechnung von d und Δd
df = pd.DataFrame(hkl_data)
df['d'] = a_0 / np.sqrt(df['h']**2 + df['k']**2 + df['l']**2)
df['d_error'] = 0.1 * df['d']  # 10% geschätzter Fehler

# Berechnung der Wellenlängen und Fehler
df['theta_rad'] = np.radians(theta_values)  # Umwandlung von Grad in Radiant

# Wellenlängenberechnung (lambda)
df["lambda"] = 2 * df['d'] * np.sin(df["theta_rad"])

# Fehlerberechnung für Wellenlänge (lambda_error)
df["lambda_error"] = np.sqrt(
    (2 * np.sin(df["theta_rad"]) * df['d_error'])**2 +
    (2 * df['d'] * np.cos(df["theta_rad"]) * delta_theta_values)**2
)

# Plot der Messpunkte mit Fehlerbalken
plt.figure(figsize=(10, 10))
plt.errorbar(x_values, y_values, xerr=delta_x, yerr=delta_y, fmt='o', color='blue', 
             ecolor='darkcyan', elinewidth=1, capsize=3, label="Messpunkte mit Fehler")
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')

# Achseneinstellungen
plt.xlabel("X-Achse (mm)", fontsize=14)
plt.ylabel("Y-Achse (mm)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color='lightgray', linestyle='--', linewidth=0.7, alpha=0.7)
plt.legend(fontsize=12)

# Achsenverhältnis (gleichmäßig)
plt.axis('equal')

# Bild speichern und anzeigen
plt.savefig(f"{path}\\messpunkte_mit_fehler_zq_theta.png", dpi=300, bbox_inches='tight')
plt.show()

# Ausgabe der Ergebnisse für Netzebenenabstände und Wellenlängen
print(df)

# Speichern als CSV-Datei
df.to_csv(f"{path}\\netzebenenabstaende_mit_lambda.csv", index=False)