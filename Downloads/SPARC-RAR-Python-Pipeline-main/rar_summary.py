from data_preprocess import load_sparc_dataset
from best_fit_calculation import fit_rar
def generate_report():
    df = load_sparc_dataset()
    g0 = fit_rar()

    print("===== SPARC SUMMARY REPORT =====")
    print("Number of galaxies:", df['ID'].nunique())
    print("Number of total points:", df.shape[0])
    print("Best-fit g0 (RAR):", g0)

    print("Report generation complete.")

if __name__ == "__main__":
    generate_report()
