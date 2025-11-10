from data_preprocess import load_sparc_dataset
import matplotlib.pyplot as plt
import numpy as np

def plot_rotation_curve(galaxy_id, output_path=None):
    df = load_sparc_dataset()
    gal = df[df['ID'] == galaxy_id].sort_values('R_kpc')

    if gal.empty:
        print(f"Galaxy '{galaxy_id}' not found.")
        return

    plt.figure(figsize=(7,5))
    plt.errorbar(gal['R_kpc'], gal['Vobs_kms'], yerr=gal['eV_kms'], fmt='o', markersize=4, label='Observed', color='blue')
    plt.plot(gal['R_kpc'], gal['Vbar_kms'], label='Baryonic Prediction', color='orange', linewidth=2)
    
    plt.title(f"Rotation Curve — {galaxy_id}")
    plt.xlabel("Radius R (kpc)")
    plt.ylabel("Velocity V (km/s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.savefig(f"{galaxy_id}_rotation_curve.png", dpi=300)
    plt.show()

def plot_rar(df=None, output_path="RAR_plot.png"):
    if df is None:
        df = load_sparc_dataset()
    
    mask = (df['g_bar'] > 0) & (df['g_obs'] > 0)
    df = df[mask]

    plt.figure(figsize=(7,6))
    plt.loglog(df['g_bar'], df['g_obs'], 'o', markersize=3, alpha=0.6, label='SPARC data')

    x = np.logspace(np.log10(df['g_bar'].min()), np.log10(df['g_bar'].max()), 100)
    y = x
    plt.loglog(x, y, 'r--', label='y = x', linewidth=1.5)

    plt.xlabel("Baryonic Acceleration g_bar (m/s²)")
    plt.ylabel("Observed Acceleration g_obs (m/s²)")
    plt.title("Radial Acceleration Relation (RAR)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_rotation_curve("UGC02953")
    plot_rar()
