"""
MAIN PIPELINE CONTROLLER FOR SPARC RAR ANALYSIS

This script runs the full project workflow:
1) Convert RTF → text → CSV (if needed)
2) Preprocess SPARC data
3) Filter high-error data
4) Plot rotation curves for selected galaxies
5) Compute and visualize RAR
6) Fit halo models (Burkert) for selected galaxies
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


def main():
    print("========================================")
    print("  SPARC RAR FULL PIPELINE STARTED")
    print("========================================")

    ensure_output_dirs()
    convert_rtf_if_needed()

    df_preprocessed = preprocess_data()
    df_filtered = filter_data()

    # --- Ask user for Galaxy IDs ---
    available_galaxies = df_filtered['ID'].unique()
    print(f">>> Available galaxies ({len(available_galaxies)}): {', '.join(available_galaxies)}")
    user_input = input("Enter Galaxy IDs to process (comma-separated): ").strip()
    galaxy_ids = [gid.strip() for gid in user_input.split(',') if gid.strip()]

    # --- Plot rotation curves for selected galaxies ---
    print(">>> Plotting rotation curves for selected galaxies...")
    for gid in galaxy_ids:
        if gid in df_filtered['ID'].values:
            plot_rotation_curve.plot_rotation_curve(gid)
        else:
            print(f"Galaxy '{gid}' not found. Skipping.")

    # --- Visualize RAR for full dataset ---
    print(">>> Creating RAR plots for full dataset...")
    visualize_rotation_rar.plot_rar(df_filtered, output_path="outputs/rar_plots/rar_full.png")

    # --- Fit Burkert halos for selected galaxies ---
    print(">>> Fitting Burkert halo profiles for selected galaxies...")
    for gid in galaxy_ids:
        if gid in df_filtered['ID'].values:
            burkert_halo_fit.fit_burkert(gid)
        else:
            print(f"Galaxy '{gid}' not found. Skipping.")

    # --- Compute best-fit g0 ---
    print(">>> Computing best-fit RAR acceleration scale g0...")
    g0 = best_fit_calculation.fit_rar()
    print(f"BEST-FIT g0 = {g0:.3e} m/s^2")

    # --- Generate final summary report ---
    print(">>> Generating final summary report...")
    rar_summary.generate_report()
    print("Final report generated")

    print("========================================")
    print("  PIPELINE COMPLETE ✅")
    print("========================================")


if __name__ == "__main__":
    main()
