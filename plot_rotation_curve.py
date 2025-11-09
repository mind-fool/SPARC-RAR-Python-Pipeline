# plot_rotation_curve.py
from data_preprocess import load_sparc_dataset
import matplotlib.pyplot as plt

def plot_rotation_curve(galaxy_id):
    df = load_sparc_dataset()
    gal = df[df['ID'] == galaxy_id].sort_values('R_kpc')

    if gal.empty:
        print("Galaxy not found.")
        return

    plt.figure(figsize=(7,5))
    plt.errorbar(gal['R_kpc'], gal['Vobs_kms'], yerr=gal['eV_kms'], fmt='o')
    plt.title(f"Rotation Curve — {galaxy_id}")
    plt.xlabel("R (kpc)")
    plt.ylabel("V_obs (km/s)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    plot_rotation_curve("UGC02953")
