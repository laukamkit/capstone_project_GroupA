import os
import glob
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

root = tk.Tk()
root.withdraw()

RESULTS_DIR = filedialog.askdirectory(title="Select the results folder")
if not RESULTS_DIR:
    messagebox.showerror("No folder selected", "No folder was selected. Exiting.")
    raise SystemExit("No folder selected.")
MASTER_DIR = filedialog.askdirectory(title="Select the master folder")
if not MASTER_DIR:
    messagebox.showerror("No folder selected", "No folder was selected. Exiting.")
    raise SystemExit("No folder selected.")
PREFIX = simpledialog.askstring("File prefix", "Enter a prefix for the output file name (leave blank for none):", initialvalue="")
PREFIX = (PREFIX.strip() + "_") if PREFIX and PREFIX.strip() else ""

# Ask user which file type to merge
file_type_window = tk.Toplevel()
file_type_window.title("Select file type")
file_type_window.geometry("320x120")
file_type_window.grab_set()
file_type_var = tk.StringVar(value="test_metrics")
tk.Label(file_type_window, text="Which CSV files do you want to merge?").pack(pady=(12, 4))
tk.Radiobutton(file_type_window, text="_test_metrics.csv", variable=file_type_var, value="test_metrics").pack(anchor="w", padx=40)
tk.Radiobutton(file_type_window, text="_fitting_log.csv", variable=file_type_var, value="fitting_log").pack(anchor="w", padx=40)
tk.Button(file_type_window, text="OK", command=file_type_window.destroy).pack(pady=8)
file_type_window.wait_window()
FILE_SUFFIX = "_test_metrics.csv" if file_type_var.get() == "test_metrics" else "_fitting_log.csv"
OUTPUT_LABEL = "master_test_metrics" if file_type_var.get() == "test_metrics" else "master_validation_fitting_log"

MASTER_FILE = os.path.join(MASTER_DIR, f"{PREFIX}{OUTPUT_LABEL}.csv")

frames = []

for filepath in glob.glob(os.path.join(RESULTS_DIR, "**", f"*{FILE_SUFFIX}"), recursive=True):
    try:
        df = pd.read_csv(filepath)
        df.insert(0, "file_name", os.path.basename(filepath))
        frames.append(df)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

if os.path.exists(MASTER_FILE):
    existing_df = pd.read_csv(MASTER_FILE)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
else:
    combined = new_df

combined.to_csv(MASTER_FILE, index=False)
print(f"Master file saved to: {MASTER_FILE} ({len(combined)} entries)")