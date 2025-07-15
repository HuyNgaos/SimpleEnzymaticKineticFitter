import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === Model definitions ===
def exp_model(t, A, k):
    return A * (1 - np.exp(-k * t))

def mm_model(S, Vmax, Km):
    return (Vmax * S) / (Km + S)

def binding_model(S, kmax, Kd):
    return (kmax * S) / (Kd + S)

# === GUI App Class ===
class FittingApp:
    def __init__(self, master):
        self.master = master
        master.title("Kinetic Curve Fitter")

        # Labels and input fields
        tk.Label(master, text="Plot Title").grid(row=0, column=0)
        tk.Label(master, text="X-axis Label").grid(row=1, column=0)
        tk.Label(master, text="Y-axis Label").grid(row=2, column=0)

        self.title_entry = tk.Entry(master)
        self.xlabel_entry = tk.Entry(master)
        self.ylabel_entry = tk.Entry(master)

        self.title_entry.insert(0, "Kinetic Fit")
        self.xlabel_entry.insert(0, "x-axis")
        self.ylabel_entry.insert(0, "y-axis")

        self.title_entry.grid(row=0, column=1)
        self.xlabel_entry.grid(row=1, column=1)
        self.ylabel_entry.grid(row=2, column=1)

        # Model selection
        tk.Label(master, text="Select Model").grid(row=3, column=0)
        self.model_var = tk.StringVar(value="Exponential")
        tk.OptionMenu(master, self.model_var,
                      "Exponential",
                      "Michaelis-Menten",
                      "k_max (k_obs vs [S])").grid(row=3, column=1) # Model options

        # Buttons
        tk.Button(master, text="Load CSV", command=self.load_data).grid(row=4, column=0, pady=10)
        tk.Button(master, text="Fit and Plot", command=self.fit_data).grid(row=4, column=1)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            self.x_data = df.iloc[:, 0].values
            self.y_data = df.iloc[:, 1].values
            messagebox.showinfo("Success", f"Loaded data from {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")

    def fit_data(self):
        try:
            model_choice = self.model_var.get()
            title_name = self.title_entry.get()

            # === match-case for model selection ===
            match model_choice:
                case "Exponential":
                    model_func = exp_model
                    p0 = [max(self.y_data), 1.0]
                    param_names = ['A', 'k']
                    eqn_str = r"$y = A(1 - e^{-kt})$"
                case "Michaelis-Menten":
                    model_func = mm_model
                    p0 = [max(self.y_data), np.median(self.x_data)]
                    param_names = ['Vmax', 'Km']
                    eqn_str = r"$v = \frac{V_{max} \cdot [S]}{K_m + [S]}$"
                case "k_max (k_obs vs [S])":
                    model_func = binding_model
                    p0 = [max(self.y_data), np.median(self.x_data)]
                    param_names = ['k_max', 'K_d']
                    eqn_str = r"$k_{obs} = \frac{k_{max} \cdot [S]}{K_d + [S]}$"
                case _:
                    raise ValueError("Invalid model selected.")

            # === Fit ===
            params, cov = curve_fit(model_func, self.x_data, self.y_data, p0=p0)
            stderr = np.sqrt(np.diag(cov))
            y_fit = model_func(self.x_data, *params)

            # R²
            residuals = self.y_data - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.y_data - np.mean(self.y_data))**2)
            r2 = 1 - ss_res / ss_tot
            
            # === Print to console ===
            print("\n========== Fit Results ==========")
            for name, val, err in zip(param_names, params, stderr):
                print(f"{name:6} = {val:.4f} ± {err:.4f}")
            print(f"R²     = {r2:.4f}")
            print("=================================\n")

            # Save results
            results = {
                "Parameter": param_names + ["R_squared"],
                "Value": list(params) + [r2],
                "Standard Error": list(stderr) + [""]
            }
            pd.DataFrame(results).to_csv("fit_results.csv", index=False)

            # === Plot ===
            x_smooth = np.linspace(min(self.x_data), max(self.x_data), 200)
            y_smooth = model_func(x_smooth, *params)

            plt.figure(figsize=(6, 4))
            plt.scatter(self.x_data, self.y_data, label=self.ylabel_entry.get(), color='red')

            legend_text = model_choice + "\n"
            for name, val, err in zip(param_names, params, stderr):
                legend_text += f"{name} = {val:.2f} ± {err:.2f}, "
            legend_text += f"$R^2$ = {r2:.3f}"

            plt.plot(x_smooth, y_smooth, label=legend_text, color='blue')
            plt.xlabel(self.xlabel_entry.get())
            plt.ylabel(self.ylabel_entry.get())
            plt.title(title_name + ": " + eqn_str)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("fit_plot.png", dpi=300)
            plt.show()

            messagebox.showinfo("Done", "Fit completed.\nSaved: fit_results.csv and fit_plot.png")

        except Exception as e:
            messagebox.showerror("Error", f"Fitting failed:\n{e}")

# === Launch the app ===

def main():
    introduction = """The non-linear fitting app for free hehe."""
    print(introduction)
    root = tk.Tk()
    app = FittingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()