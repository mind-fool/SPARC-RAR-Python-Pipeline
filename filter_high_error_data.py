from data_preprocess import load_sparc_dataset

def filter_data(threshold=0.2, *, df=None, output_path="filtered_data.csv"):
    if df is None:
        df = load_sparc_dataset()
    df = df.assign(rel_err=df["eV_kms"] / df["Vobs_kms"])

    filtered = df[df['rel_err'] < threshold]
    filtered.to_csv(output_path, index=False)
    print(f"Saved {len(filtered)} rows to {output_path}")
    return filtered

if __name__ == "__main__":
    filter_data()
