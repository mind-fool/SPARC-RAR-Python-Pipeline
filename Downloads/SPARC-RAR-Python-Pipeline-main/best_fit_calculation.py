from scipy.optimize import least_squares 
from data_preprocess import load_sparc_dataset
import numpy as np

def rar_model(gbar, g0):
    return gbar / (1 - np.exp(-np.sqrt(gbar / g0)))

def fit_rar():
    df = load_sparc_dataset()
    mask = (df['g_bar'] > 0) & (df['g_obs'] > 0)
    gbar = df[mask]['g_bar'].values
    gobs = df[mask]['g_obs'].values
    gobs_err = df[mask]['g_obs_err'].values

    # fallback weights
    if np.any(np.isnan(gobs_err)) or np.all(gobs_err == 0):
        gobs_err = np.maximum(0.1*gobs, 1e-12)

    def residuals(p):
        g0 = p[0]
        return (gobs - rar_model(gbar, g0)) / gobs_err

    p0 = [1e-10]
    result = least_squares(residuals, p0, bounds=(1e-16, 1e-5))
    g0_fit = result.x[0]

    print(f"Best-fit g0: {g0_fit:.3e} m/s^2")

    return g0_fit

if __name__ == "__main__":
    fit_rar()
