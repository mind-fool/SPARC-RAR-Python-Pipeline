import pandas as pd
import numpy as np

def load_sparc_dataset(path="GALAXY_CSVDATA.csv"):
    """
    Loads and prepares SPARC-style galaxy rotation curve data.
    Returns a processed DataFrame with accelerations computed.
    """
    df = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)

    colmap = {
        'ID': 'ID',
        'D(Assumed_Distance)': 'D_Mpc',
        'R(Galactocentric_Rad...)': 'R_kpc',
        'Vobs': 'Vobs_kms',
        'e_Vbos': 'eV_kms',
        'Vgas': 'Vgas_kms',
        'Vdisk': 'Vdisk_kms',
        'Vbul': 'Vbul_kms',
        'SBdisk': 'SBdisk',
        'SBbul': 'SBb'
    }
    df = df.rename(columns=colmap)

    """***Numeric columns***"""
    numeric_cols = ['D_Mpc','R_kpc','Vobs_kms','eV_kms','Vgas_kms','Vdisk_kms','Vbul_kms']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    """***Unit conversion constants***"""
    km_to_m = 1000.0
    kpc_to_m = 3.085677581e19

    """***Convert to SI***"""
    df['Vobs_ms'] = df['Vobs_kms'] * km_to_m
    df['Vgas_ms'] = df['Vgas_kms'] * km_to_m
    df['Vdisk_ms'] = df['Vdisk_kms'] * km_to_m
    df['Vbul_ms'] = df['Vbul_kms'] * km_to_m
    df['R_m'] = df['R_kpc'] * kpc_to_m

    """***Baryonic velocity prediction***"""
    df['Vbar_kms'] = np.sqrt(
        df['Vgas_kms']**2 +
        df['Vdisk_kms']**2 +
        df['Vbul_kms']**2
    )
    df['Vbar_ms'] = df['Vbar_kms'] * km_to_m

    """***Accelerations***"""
    df['g_obs'] = df['Vobs_ms']**2 / df['R_m']
    df['g_bar'] = df['Vbar_ms']**2 / df['R_m']

    """***Uncertainty propagation***"""
    df['eV_ms'] = df['eV_kms'] * km_to_m
    df['g_obs_err'] = 2 * df['Vobs_ms'] * df['eV_ms'] / df['R_m']

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df

if __name__ == "__main__":
    df = load_sparc_dataset()
    print(df.head())