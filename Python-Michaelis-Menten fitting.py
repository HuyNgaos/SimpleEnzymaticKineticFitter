import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# === 0. Inputs ===
# Replace these with your actual data
substrate_data = [1, 2, 5, 10, 20, 50, 100]  # [S]
velocity_data = [0.09, 0.16, 0.29, 0.45, 0.6, 0.8, 0.88]  # v
data_point_label = "v (nM/s)"  # Label for the dot legend
fit_label = f"Michaelis-Menten Fit"  # Label for the fit line legend
x_axis_label = "Substrate [S] (nM)"  # X-axis label
y_axis_label = "Velocity v (nM/s)"  # Y-axis label

# === 1. Michaelis-Menten model === replace with your model function if needed
def michaelis_menten(S, Vmax, Km):
    return (Vmax * S) / (Km + S)

# === 2. Input data (replace with your real data) ===
substrate = np.array(substrate_data)  # [S]
velocity = np.array(velocity_data)  # v

# === 3. Fit === #change the parameters as needed
initial_guess = [max(velocity), np.median(substrate)]
params, covariance = curve_fit(michaelis_menten, substrate, velocity, p0=initial_guess)
Vmax_fit, Km_fit = params
stderr = np.sqrt(np.diag(covariance))
Vmax_err, Km_err = stderr

# === 4. R² === #change the parameters as needed
v_fit = michaelis_menten(substrate, Vmax_fit, Km_fit)
residuals = velocity - v_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((velocity - np.mean(velocity))**2)
r_squared = 1 - (ss_res / ss_tot)

# === 5. Print output === #change the parameters as needed
print(f"Fitted Vmax = {Vmax_fit:.4f} ± {Vmax_err:.4f}")
print(f"Fitted Km    = {Km_fit:.4f} ± {Km_err:.4f}")
print(f"R²           = {r_squared:.4f}")

# === 6. Save CSV === #change the parameters as needed
results = {
    "Parameter": ["Vmax", "Km", "R_squared"],
    "Value": [Vmax_fit, Km_fit, r_squared],
    "Standard Error": [Vmax_err, Km_err, ""]
}
df = pd.DataFrame(results)
df.to_csv("MM_fit_results.csv", index=False)
print("CSV file saved as 'MM_fit_results.csv'.")

# === 7. Plot ===
S_smooth = np.linspace(0, max(substrate) * 1.1, 200)
v_smooth = michaelis_menten(S_smooth, Vmax_fit, Km_fit)

plt.figure(figsize=(6, 4))
plt.scatter(substrate, velocity, color='red', label=data_point_label)
plt.plot(S_smooth, v_smooth, color='blue', label=fit_label)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)
plt.title(r"Michaelis-Menten Fit: $v = \frac{V_{max} \cdot [S]}{K_m + [S]}$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("MM_fit_plot.png", dpi=300)
plt.close()
print("Plot saved as 'MM_fit_plot.png'.")
