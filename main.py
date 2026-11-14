"""Run the SPARC analysis workflows from one entry point."""

from pathlib import Path

from best_fit_calculation import fit_rar
from burkert_halo_fit import DEFAULT_MAX_SCALE_RADIUS_KPC
from halo_fit_selection import fit_best_halo
from compare_halo_fits import plot_halo_comparison
from data_preprocess import load_sparc_dataset
from filter_high_error_data import filter_data
from plot_rotation_curve import plot_rotation_curve
from rar_summary import generate_report
from report_converter import convert_rtf_to_text, convert_text_to_csv
from run_mass_to_light_sweep import main as run_mass_to_light_sweep
from run_nfw_bound_sweep import main as run_nfw_bound_sweep
from run_boundary_sensitivity import run_boundary_sensitivity
from visualize_rotation_rar import plot_rar
from run_optimizer_validation import run_validation


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "GALAXY_CSVDATA.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def prepare_input():
    """Generate the input CSV on a fresh checkout."""
    if DATA_PATH.is_file():
        return
    text_path = PROJECT_DIR / "Galaxy_textfile.txt"
    if not text_path.is_file():
        result = convert_rtf_to_text(PROJECT_DIR / "Galaxy_textfile.rtf", text_path)
        if result is None:
            raise SystemExit("Could not convert the bundled SPARC report.")
    convert_text_to_csv(text_path, DATA_PATH)


def run_rar_workflow(data):
    """Fit, plot and summarize the same filtered acceleration data."""
    filtered = filter_data(df=data, output_path=OUTPUT_DIR / "filtered_data.csv")
    g0 = fit_rar(filtered)
    plot_rar(filtered, output_path=OUTPUT_DIR / "rar_plots" / "rar_full.png",
             g0=g0, show=False)
    generate_report(df=filtered, g0=g0)


def run_galaxy_workflow(data):
    """Save the baryonic curve, both halo fits and the comparison for each ID."""
    available = {name.casefold(): name for name in data["ID"].unique()}
    requested = input("Enter galaxy IDs separated by commas [NGC0100]: ").strip()
    galaxy_ids = []
    for name in (requested or "NGC0100").split(","):
        name = name.strip()
        if not name:
            continue
        galaxy_id = available.get(name.casefold())
        if galaxy_id is None:
            print(f"Galaxy {name!r} was not found in the preprocessed sample.")
        elif galaxy_id not in galaxy_ids:
            galaxy_ids.append(galaxy_id)
    if not galaxy_ids:
        return

    value = input(f"NFW upper scale radius in kpc [{DEFAULT_MAX_SCALE_RADIUS_KPC:g}]: ").strip()
    try:
        cap = float(value) if value else DEFAULT_MAX_SCALE_RADIUS_KPC
        if not 0.01 < cap < float("inf"):
            raise ValueError
    except ValueError:
        print("Enter a finite NFW upper scale radius greater than 0.01 kpc.")
        return
    suffix = "" if cap == DEFAULT_MAX_SCALE_RADIUS_KPC else f"_cap{cap:g}kpc"
    halo_dir = OUTPUT_DIR / "halo_fits"
    for galaxy_id in galaxy_ids:
        try:
            burkert = fit_best_halo(galaxy_id, df=data, model="burkert", show=False, plot=True, report=True,
                              output_path=halo_dir / f"{galaxy_id}_burkert_fit.png")
            nfw = fit_best_halo(galaxy_id, df=data, model="nfw", show=False, plot=True, report=True,
                          max_scale_radius_kpc=cap,
                          output_path=halo_dir / f"{galaxy_id}_nfw{suffix}_fit.png")
        except (ValueError, RuntimeError) as error:
            print(f"Could not complete halo fits for {galaxy_id}: {error}")
            continue
        plot_rotation_curve(galaxy_id, df=data, show=False,
                            output_path=OUTPUT_DIR / "rotation_curves" / f"{galaxy_id}_rotation_curve.png")
        plot_halo_comparison(galaxy_id, burkert, nfw, show=False,
                             output_path=halo_dir / f"{galaxy_id}_halo_comparison{suffix}.png")


def main():
    print("SPARC analysis")
    print("1. Radial acceleration relation (RAR)")
    print("2. Rotation curves and halo comparisons for selected galaxies")
    print("3. NFW radius-limit sweep (175-galaxy study)")
    print("4. Stellar mass-to-light sweep")
    print("5. Run all analyses")
    print("6. Check optimizer starting-point sensitivity")
    print("7. Test halo-fit boundary sensitivity (full 175-galaxy study)")
    print("0. Quit")
    choice = input("Choose an analysis [5]: ").strip() or "5"
    if choice == "0":
        return
    if choice not in {"1", "2", "3", "4", "5", "6", "7"}:
        raise SystemExit("Choose a number from 0 to 7.")

    prepare_input()
    for folder in ("rotation_curves", "rar_plots", "halo_fits", "sweeps"):
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    if choice in {"1", "2", "5"}:
        data = load_sparc_dataset(DATA_PATH)
        data.to_csv(OUTPUT_DIR / "preprocessed_data.csv", index=False)
        if choice in {"2", "5"}:
            run_galaxy_workflow(data)
        if choice in {"1", "5"}:
            run_rar_workflow(data)

    if choice in {"3", "5"}:
        run_nfw_bound_sweep(output_path=OUTPUT_DIR / "sweeps" / "nfw_bound_sweep.csv")
    if choice in {"4", "5"}:
        run_mass_to_light_sweep(output_path=OUTPUT_DIR / "sweeps" / "mass_to_light_sweep.csv")

    if choice in {"5", "6"}:
        selected = None
        if choice == "6":
            requested = input("Galaxy IDs separated by commas [all galaxies]: ").strip()
            selected = requested.split(",") if requested else None
        run_validation(output_dir=OUTPUT_DIR / "validation", galaxy_ids=selected,
                       data_path=DATA_PATH)

    if choice == "7":
        run_boundary_sensitivity(
            output_dir=OUTPUT_DIR / "boundary_sensitivity", data_path=DATA_PATH,
        )

    print(f"Results saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
