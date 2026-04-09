import os
import glob
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

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
MASTER_FILE = os.path.join(MASTER_DIR, "master_metrics.csv")

records = []

for filepath in glob.glob(os.path.join(RESULTS_DIR, "**", "*_test_results.csv"), recursive=True):
    try:
        row = pd.read_csv(filepath, nrows=1)
        if "rmse" not in row.columns or "mae" not in row.columns:
            print(f"Skipping (missing columns): {filepath}")
            continue
        records.append({
            "file_name": os.path.basename(filepath),
            "rmse": row["rmse"].iloc[0],
            "mae": row["mae"].iloc[0],
        })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

new_df = pd.DataFrame(records, columns=["file_name", "rmse", "mae"])

if os.path.exists(MASTER_FILE):
    existing_df = pd.read_csv(MASTER_FILE)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["file_name"], keep="last")
else:
    combined = new_df

combined.to_csv(MASTER_FILE, index=False)
print(f"Master file saved to: {MASTER_FILE} ({len(combined)} entries)")