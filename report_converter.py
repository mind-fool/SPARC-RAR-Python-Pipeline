import pandas as pd
import io

from striprtf.striprtf import rtf_to_text
with open('Galaxy_textfile.rtf', 'r') as file:
    rtf_content = file.read()

plain_text_content = rtf_to_text(rtf_content)

print("RTF content successfully read and converted to plain text.")

data_io = io.StringIO(plain_text_content)
df = pd.read_csv(data_io, sep='\s+', engine='python')

print("DataFrame created successfully:")
print(df)

output_csv_file = "GALAXY_CSVDATA.csv"
df.to_csv(output_csv_file, index=False)

print(f"DataFrame successfully saved to {output_csv_file}")