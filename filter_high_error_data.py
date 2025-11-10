from data_preprocess import load_sparc_dataset

def filter_data(threshold=0.2):
    df = load_sparc_dataset()
    df['rel_err'] = df['eV_kms'] / df['Vobs_kms']

    filtered = df[df['rel_err'] < threshold]
    filtered.to_csv("filtered_data.csv", index=False)
    print(f"Saved {filtered.shape[0]} rows to filtered_data.csv")
    return filtered

if __name__ == "__main__":
    filter_data()
