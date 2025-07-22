import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re

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
        self.df = None
        self.y_column_index = None
        self.table_window = None

        # Input fields
        tk.Label(master, text="Plot Title").grid(row=0, column=0)
        tk.Label(master, text="X-axis Label").grid(row=1, column=0)
        tk.Label(master, text="Y-axis Label").grid(row=2, column=0)
        tk.Label(master, text="k Initial Guess (e.g. k, Km, Kd)").grid(row=3, column=0)

        self.title_entry = tk.Entry(master)
        self.xlabel_entry = tk.Entry(master)
        self.ylabel_entry = tk.Entry(master)
        self.init_guess_entry = tk.Entry(master)

        self.title_entry.insert(0, "Kinetic Fit")
        self.xlabel_entry.insert(0, "Time")
        self.ylabel_entry.insert(0, "%Products")
        self.init_guess_entry.insert(0, "1")

        self.title_entry.grid(row=0, column=1)
        self.xlabel_entry.grid(row=1, column=1)
        self.ylabel_entry.grid(row=2, column=1)
        self.init_guess_entry.grid(row=3, column=1)

        # Model selection
        tk.Label(master, text="Select Model").grid(row=4, column=0)
        self.model_var = tk.StringVar(value="Exponential")
        tk.OptionMenu(master, self.model_var,
                      "Exponential",
                      "Michaelis-Menten",
                      "k_obs vs [S]").grid(row=4, column=1)

        # Checkboxes
        self.include_fit_var = tk.BooleanVar(value=True)
        self.show_grid_var = tk.BooleanVar(value=True)

        tk.Checkbutton(master, text="Include fit results in legend", variable=self.include_fit_var)\
            .grid(row=5, column=0, columnspan=2)
        tk.Checkbutton(master, text="Show grid on plot", variable=self.show_grid_var)\
            .grid(row=6, column=0, columnspan=2, pady=(0, 10))

        # Buttons
        tk.Button(master, text="Load CSV", command=self.load_data).grid(row=7, column=0, pady=10)
        tk.Button(master, text="Fit and Plot", command=self.fit_data).grid(row=7, column=1)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            self.df = pd.read_csv(file_path)
            self.show_table_window()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")

    def show_table_window(self):
        if self.df is None:
            return

        if self.table_window is not None and self.table_window.winfo_exists():
            self.table_window.destroy()
            messagebox.showinfo("Old Table Closed", "Previous data table window was closed.")

        self.y_column_index = tk.IntVar(value=1)
        self.table_window = tk.Toplevel(self.master)
        self.table_window.title("Select Y Column")

        # === Set minimum size to 3 cm width (≈113 px)
        min_width_px = 202
        self.table_window.update_idletasks()
        self.table_window.minsize(min_width_px, 200)  # height is flexible

        for j in range(1, len(self.df.columns)):
            btn = tk.Radiobutton(self.table_window, variable=self.y_column_index, value=j)
            btn.grid(row=0, column=j, sticky="nsew", padx=1, pady=1)

        for j, col_name in enumerate(self.df.columns):
            e = tk.Entry(self.table_window, width=12, justify='center', bg='lightblue')
            e.insert(0, col_name)
            e.config(state='readonly')
            e.grid(row=1, column=j, padx=1, pady=1)

        for i in range(len(self.df)):
            for j in range(len(self.df.columns)):
                e = tk.Entry(self.table_window, width=12, justify='center')
                e.insert(0, str(self.df.iat[i, j]))
                e.config(state='readonly')
                e.grid(row=i+2, column=j, padx=1, pady=1)


    def sanitize_filename(self, name):
        return re.sub(r'\W+', '_', name.strip())

    def fit_data(self):
        try:
            if self.df is None:
                messagebox.showwarning("No Data", "Please load a CSV file first.")
                return

            x_data = self.df.iloc[:, 0].values
            y_col_idx = self.y_column_index.get()
            y_data = self.df.iloc[:, y_col_idx].values

            x_header_raw = self.df.columns[0]
            y_header_raw = self.df.columns[y_col_idx]
            x_header = self.sanitize_filename(x_header_raw)
            y_header = self.sanitize_filename(y_header_raw)

            model_choice = self.model_var.get()

            match model_choice:
                case "Exponential":
                    model_func = exp_model
                    param_names = ['A', 'k']
                    eqn_str = r"$y = A(1 - e^{-kt})$"
                    model_suffix = "Exponential"
                case "Michaelis-Menten":
                    model_func = mm_model
                    param_names = ['Vmax', 'Km']
                    eqn_str = r"$v = \frac{V_{max} \cdot [S]}{K_m + [S]}$"
                    model_suffix = "Michaelis_Menten"
                case "k_obs vs [S]":
                    model_func = binding_model
                    param_names = ['k_max', 'K_d']
                    eqn_str = r"$k_{obs} = \frac{k_{max} \cdot [S]}{K_d + [S]}$"
                    model_suffix = "Binding"
                case _:
                    raise ValueError("Invalid model selected.")

            filename_base = f"{y_header}_{x_header}_fitting_{model_suffix}"

            try:
                user_guess = float(self.init_guess_entry.get())
            except ValueError:
                messagebox.showwarning("Invalid Input", "k initial guess must be a number. Defaulting to 1.")
                user_guess = 1.0

            p0 = [max(y_data), user_guess]

            params, cov = curve_fit(model_func, x_data, y_data, p0=p0)
            stderr = np.sqrt(np.diag(cov))
            y_fit = model_func(x_data, *params)

            residuals = y_data - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2 = 1 - ss_res / ss_tot

            print("\n========== Fit Results ==========")
            for name, val, err in zip(param_names, params, stderr):
                print(f"{name:6} = {val:.4f} ± {err:.4f}")
            print(f"R²     = {r2:.4f}")
            print("=================================\n")

            results = {
                "Parameter": param_names + ["R_squared"],
                "Value": list(params) + [r2],
                "Standard Error": list(stderr) + [""]
            }

            pd.DataFrame(results).to_csv(f"{filename_base}.csv", index=False)

            x_smooth = np.linspace(min(x_data), max(x_data), 200)
            y_smooth = model_func(x_smooth, *params)

            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, label=self.ylabel_entry.get(), color='red')

            if self.include_fit_var.get():
                legend_text = model_choice + "\n"
                for name, val, err in zip(param_names, params, stderr):
                    legend_text += f"{name} = {val:.2f} ± {err:.2f}, "
                legend_text += f"$R^2$ = {r2:.3f}"
            else:
                legend_text = model_choice

            plt.plot(x_smooth, y_smooth, label=legend_text, color='blue')
            plt.xlabel(self.xlabel_entry.get())
            plt.ylabel(self.ylabel_entry.get())
            plt.title(self.title_entry.get() + ": " + eqn_str)
            plt.legend()
            plt.grid(self.show_grid_var.get())
            plt.tight_layout()
            plt.savefig(f"{filename_base}.png", dpi=300)
            plt.show()

            messagebox.showinfo("Done", f"Fit completed.\nSaved:\n{filename_base}.csv\n{filename_base}.png")

        except Exception as e:
            messagebox.showerror("Error", f"Fitting failed:\n{e}")

# === Launch ===
def main():
    print("The non-linear fitting app for free hehe.")
    root = tk.Tk()
    app = FittingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
