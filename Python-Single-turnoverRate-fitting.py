import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# === 0. Inputs ===
# Replace these with your actual data
time_data = [1, 2, 5, 10, 20, 50, 100]  # [S]
kobs_data = [0.09, 0.16, 0.29, 0.45, 0.6, 0.8, 0.88]  # v
data_point_label = "kobs"  # Label for the dot legen
fit_label = f"First-order exponential Fit"  # Label for the fit line legend
x_axis_label = "Substrate [S] (nM)"  # X-axis label
y_axis_label = "kobs"  # Y-axis label

# === 1. Define model function ===
def model(t, A, k):
    return A * (1 - np.exp(-k * t))

# === 2. Input data (replace with your actual values) ===
t_data = np.array(time_data)         # time
y_data = np.array(kobs_data)  # observed y (k_obs)

# === 3. Fit the data ===
initial_guess = [max(y_data), 1.0]
params, covariance = curve_fit(model, t_data, y_data, p0=initial_guess)
A_fit, k_fit = params
stderr = np.sqrt(np.diag(covariance))
A_err, k_err = stderr

# === 4. Compute R² ===
y_fit = model(t_data, A_fit, k_fit)
residuals = y_data - y_fit
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)

# === 5. Print results ===
print(f"Fitted A = {A_fit:.4f} ± {A_err:.4f}")
print(f"Fitted k = {k_fit:.4f} ± {k_err:.4f}")
print(f"R²       = {r_squared:.4f}")

# === 6. Save results to CSV ===
results = {
    "Parameter": ["A", "k", "R_squared"],
    "Value": [A_fit, k_fit, r_squared],
    "Standard Error": [A_err, k_err, ""]
}
df = pd.DataFrame(results)
df.to_csv("Single_turnover_fit_results.csv", index=False)
print("CSV file saved as 'Single_turnover_fit_results.csv'.")

# === 7. Plot and save ===
t_smooth = np.linspace(min(t_data), max(t_data), 200)
y_smooth = model(t_smooth, A_fit, k_fit)

plt.figure(figsize=(6, 4))
plt.scatter(t_data, y_data, label=data_point_label, color='red')
plt.plot(t_smooth, y_smooth, label=fit_label, color='blue')
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)  # LaTeX subscript formatting
plt.title(r"Fit: $y = A(1 - e^{-kt})$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Single_turnover_plot.png", dpi=300)
plt.close()
print("Plot saved as 'Single_turnover_plot.png'.")
