import pandas as pd
import io
from striprtf.striprtf import rtf_to_text

def convert_rtf_to_text(input_rtf_path, output_csv_path):
    """
    Reads an RTF file, converts it to plain text, converts it into a DataFrame,
    and saves the result as CSV.
    """

    try:
        # Read RTF file
        with open(input_rtf_path, 'r') as file:
            rtf_content = file.read()

        # Convert RTF → plain text
        plain_text_content = rtf_to_text(rtf_content)
        print("RTF content successfully read and converted to plain text.")

        # Convert plain text → DataFrame
        data_io = io.StringIO(plain_text_content)
        df = pd.read_csv(data_io, sep=r'\s+', engine='python')

        print("DataFrame created successfully:")
        print(df)

        # Save CSV
        df.to_csv(output_csv_path, index=False)
        print(f"DataFrame successfully saved to {output_csv_path}")

        return df

    except Exception as e:
        print("Error in convert_rtf_to_text:", e)
        return None
