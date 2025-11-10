from data_preprocess import load_sparc_dataset
import matplotlib.pyplot as plt

def plot_rotation_curve(galaxy_id, df=None, output_path=None):
    """
    Plots the rotation curve for a galaxy.
    df: optional preloaded DataFrame
    output_path: optional path to save figure
    """
    if df is None:
        df = load_sparc_dataset()
    
    gal = df[df['ID'] == galaxy_id].sort_values('R_kpc')

    if gal.empty:
        print("Galaxy not found.")
        return

    plt.figure(figsize=(7,5))
    plt.errorbar(gal['R_kpc'], gal['Vobs_kms'], yerr=gal['eV_kms'], fmt='o', label='Observed')
    plt.plot(gal['R_kpc'], gal['Vbar_kms'], label='Baryonic Prediction', color='orange', linewidth=2)
    plt.title(f"Rotation Curve — {galaxy_id}")
    plt.xlabel("R (kpc)")
    plt.ylabel("V (km/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved rotation curve to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_rotation_curve("UGC02953")
