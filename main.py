from data_preprocess import load_sparc_dataset
from plot_rotation_curve import plot_rotation_curve, plot_rar
from quality_cut import apply_quality_cut
from halo_fit import fit_burkert
from sparc_summary_report import generate_report

def run_pipeline(galaxy_id="UGC02953", quality_threshold=0.2):
    df = load_sparc_dataset()
    apply_quality_cut(threshold=quality_threshold)
    plot_rotation_curve(galaxy_id)
    plot_rar()
    fit_burkert(galaxy_id)
    generate_report()

if __name__ == "__main__":
    run_pipeline()
