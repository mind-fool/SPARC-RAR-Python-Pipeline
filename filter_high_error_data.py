from data_preprocess import load_sparc_dataset
def apply_quality_cut(threshold=0.2):
    df = load_sparc_dataset()
    df['rel_err'] = df['eV_kms'] / df['Vobs_kms']

    filtered = df[df['rel_err'] < threshold]

    filtered.to_csv("filtered_data.csv", index=False)
    print(f"Saved {filtered.shape[0]} rows to filtered_data.csv")

if __name__ == "__main__":
    apply_quality_cut()