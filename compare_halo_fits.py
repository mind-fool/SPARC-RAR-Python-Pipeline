import argparse

import matplotlib.pyplot as plt

from burkert_halo_fit import fit_halo
from data_preprocess import load_sparc_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Compare Burkert and NFW fits for one SPARC galaxy."
    )
    parser.add_argument("galaxy_id", nargs="?", help="SPARC galaxy ID")
    args = parser.parse_args()

    requested_id = args.galaxy_id or input("Enter a SPARC galaxy ID (e.g., NGC0100): ")
    requested_id = requested_id.strip()

    df = load_sparc_dataset()
    available_ids = {name.casefold(): name for name in df["ID"].unique()}
    galaxy_id = available_ids.get(requested_id.casefold())

    if galaxy_id is None:
        parser.error(f"Galaxy {requested_id!r} was not found in SPARC data.")

    burkert = fit_halo(galaxy_id, df=df, model="burkert", plot=False)
    nfw = fit_halo(galaxy_id, df=df, model="nfw", plot=False)

    fig, (curve_ax, residual_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6.5),
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    curve_ax.errorbar(
        burkert["radius_kpc"], burkert["observed_kms"],
        yerr=burkert["error_kms"], fmt="o", markersize=4, capsize=2,
        color="0.25", label="SPARC measurements",
    )

    for label, fit, style in (
        ("Burkert", burkert, "-"),
        ("NFW", nfw, "--"),
    ):
        line, = curve_ax.plot(
            fit["radius_kpc"], fit["model_kms"], linestyle=style,
            linewidth=2, label=f"Baryons + {label}",
        )
        residual_ax.plot(
            fit["radius_kpc"], fit["residuals_kms"], linestyle=style,
            marker=".", color=line.get_color(),
        )

    curve_ax.set_title(f"{galaxy_id} rotation curve")
    curve_ax.set_ylabel("Circular speed (km/s)")
    curve_ax.legend()
    residual_ax.axhline(0, color="0.4", linewidth=1, zorder=0)
    residual_ax.set_xlabel("Radius (kpc)")
    residual_ax.set_ylabel(r"$V_{\rm obs} - V_{\rm model}$ (km/s)")
    curve_ax.grid(alpha=0.25)
    residual_ax.grid(alpha=0.25)

    output = f"{galaxy_id}_halo_comparison.png"
    fig.savefig(output, dpi=300)
    print(f"Saved {output}")
    plt.show()


if __name__ == "__main__":
    main()