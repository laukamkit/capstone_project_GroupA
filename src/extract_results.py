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
MASTER_FILE = os.path.join(MASTER_DIR, f"{PREFIX}master_metrics.csv")

frames = []

for filepath in glob.glob(os.path.join(RESULTS_DIR, "**", "*_test_metrics.csv"), recursive=True):
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