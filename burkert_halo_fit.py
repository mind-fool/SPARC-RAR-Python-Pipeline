
from scipy.optimize import curve_fit
from data_preprocess import load_sparc_dataset
import numpy as np
import matplotlib.pyplot as plt

def burkert_profile(r, rho0, r0):
    """
    Burkert halo:
    V_halo^2 = 4*pi*G*rho0*r0^3 * [ln(1+r/r0) - arctan(r/r0) + 0.5 ln(1+(r/r0)^2)] / r
    """
    G = 6.67430e-11
    r = r * 3.085677581e19  # convert kpc→meters inside fit
    term = np.log(1 + r/r0) - np.arctan(r/r0) + 0.5*np.log(1 + (r/r0)**2)
    return np.sqrt(4*np.pi*G*rho0*r0**3 * term / r)

def fit_burkert(galaxy_id):
    df = load_sparc_dataset()
    gal = df[df['ID'] == galaxy_id].sort_values("R_kpc")

    R = gal['R_kpc'].values
    Vobs = gal['Vobs_kms'].values

    p0 = [1e-21, 1e20]  # initial guess
    popt, pcov = curve_fit(burkert_profile, R, Vobs, p0=p0)

    rho0, r0 = popt
    print("Best-fit halo parameters:")
    print("rho0 =", rho0)
    print("r0 = ", r0)

    plt.plot(R, Vobs, 'o')
    plt.plot(R, burkert_profile(R, *popt))
    plt.title(f"{galaxy_id} — Burkert halo fit")
    plt.xlabel("R (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.grid(True)
    plt.savefig(f"{galaxy_id}_burkert_fit.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    fit_burkert("UGC02953")
