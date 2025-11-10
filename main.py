"""
MAIN PIPELINE CONTROLLER FOR SPARC RAR ANALYSIS

This script runs the full project workflow:
1) Convert RTF → text → CSV (if needed)
2) Preprocess SPARC data
3) Filter high-error data
4) Plot rotation curves for all galaxies
5) Compute and visualize RAR
6) Fit halo models (Burkert)
7) Compute best-fit RAR acceleration scale g0
8) Generate final summary report

Author: Nirajan Rimal
"""

import os
import pandas as pd

# Import project modules
import report_converter
import data_preprocess
import filter_high_error_data
import plot_rotation_curve
import visualize_rotation_rar
import burkert_halo_fit
import best_fit_calculation
import rar_summary


def ensure_output_dirs():
    """Create folders if they don't exist."""
    dirs = [
        'outputs',
        'outputs/rotation_curves',
        'outputs/rar_plots',
        'outputs/halo_fits'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def convert_rtf_if_needed():
    """Convert the raw RTF report into usable text + CSV."""
    if not os.path.exists("Galaxy_textfile.txt"):
        print(">>> Converting RTF to text...")
        report_converter.convert_rtf_to_text("Galaxy_textfile.rtf", "Galaxy_textfile.txt")
    
    if not os.path.exists("GALAXY_CSVDATA.csv"):
        print(">>> Converting raw text to CSV...")
        report_converter.convert_text_to_csv("Galaxy_textfile.txt", "GALAXY_CSVDATA.csv")


def preprocess_data():
    """Run preprocessing and return DataFrame."""
    print(">>> Preprocessing SPARC dataset...")
    df = data_preprocess.process_sparc_dataset("GALAXY_CSVDATA.csv")
    df.to_csv("outputs/preprocessed_data.csv", index=False)
    return df


def filter_data():
    """Apply relative-error filtering."""
    print(">>> Filtering high-error data...")
    from filter_high_error_data import filter_data
    filter_data(threshold=0.2)
    df_filtered = pd.read_csv("filtered_data.csv")
    return df_filtered

# Instead of calling a non-existent function in report_converter.py
print(">>> Converting RTF report if needed...")
user_input = input("Enter the Galaxy ID you want to plot: ").strip()
    
if user_input not in data_preprocess.load_sparc_dataset()["ID"].values:
        print(f"Galaxy '{user_input}' not found. Exiting.")
import report_converter

def plot_all_rotation_curves(df):
    """Plot rotation curve for a single galaxy chosen by the user."""
    galaxies = df["ID"].unique()
    print(f">>> Available galaxies: {len(galaxies)}")
    return plot_rotation_curve.plot_rotation_curve(user_input)

def visualize_rar(df):
    """Generate RAR visualizations."""
    print(">>> Creating RAR plots...")
    visualize_rotation_rar.plot_rar(df, output_path="outputs/rar_plots/rar_full.png")


def fit_burkert_for_all(df):
    """Fit Burkert halo for the user-selected galaxy only."""
    print(f">>> Fitting Burkert halo profile for galaxy '{user_input}'...")
    burkert_halo_fit.fit_burkert(user_input)

def compute_g0(df):
    """Compute best-fit RAR acceleration scale."""
    print(">>> Performing best-fit calculation for RAR...")
    g0 = best_fit_calculation.fit_rar()
    print(f"BEST-FIT g0 = {g0:.3e} m/s^2")
    return g0

def generate_final_report(df, g0):
    """Generate summary."""
    print(">>> Generating summary report...")
    import rar_summary
    rar_summary.generate_report()
    print("Final report generated")

def main():
    print("========================================")
    print("  SPARC RAR FULL PIPELINE STARTED")
    print("========================================")

    ensure_output_dirs()
    convert_rtf_if_needed()

    df_preprocessed = preprocess_data()
    df_filtered = filter_data()

    plot_all_rotation_curves(df_filtered)
    visualize_rar(df_filtered)
    fit_burkert_for_all(df_filtered)

    g0 = compute_g0(df_filtered)
    generate_final_report(df_filtered, g0)

    print("========================================")
    print("  PIPELINE COMPLETE ✅")
    print("========================================")


if __name__ == "__main__":
    main()
