import pandas as pd
import io
from striprtf.striprtf import rtf_to_text
import os

def convert_rtf_to_text(input_rtf_path="Galaxy_textfile.rtf", output_txt_path="Galaxy_textfile.txt"):
    """
    Reads an RTF file, converts it to plain text, and saves it as a .txt file.
    """
    try:
        if not os.path.exists(input_rtf_path):
            raise FileNotFoundError(f"Input file '{input_rtf_path}' not found.")

        with open(input_rtf_path, 'r') as file:
            rtf_content = file.read()

        plain_text_content = rtf_to_text(rtf_content)

        # Save plain text to file
        with open(output_txt_path, 'w') as f:
            f.write(plain_text_content)

        print(f"RTF successfully converted to text and saved at '{output_txt_path}'")
        return plain_text_content

    except Exception as e:
        print("Error in convert_rtf_to_text:", e)
        return None


def convert_text_to_csv(input_text_path="Galaxy_textfile.txt", output_csv_path="GALAXY_CSVDATA.csv"):
    if not os.path.exists(input_text_path):
        raise FileNotFoundError(f"{input_text_path} not found")

    # Read the file
    with open(input_text_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # remove empty lines

    clean_text = "\n".join(lines)

    # Load into DataFrame
    df = pd.read_csv(io.StringIO(clean_text), delim_whitespace=True)

    print("DataFrame created successfully:")
    print(df.head())

    # Save CSV
    df.to_csv(output_csv_path, index=False)
    print(f"CSV saved to '{output_csv_path}'")

    return df

if __name__ == "__main__":
    # Example usage
    convert_rtf_to_text()
    convert_text_to_csv()
