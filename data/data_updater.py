import pandas as pd
import re


input_files = ["SwissProt-EC-dev.csv", "SwissProt-EC-test.csv", "SwissProt-EC-train.csv"]


def extract_main_ec(value):
    value_str = str(value)
    
    match = re.search(r"EC:(\d+)", value_str)
    
    if match:
        return match.group(1)
    
    return value
    

for input_file in input_files:
    df = pd.read_csv(input_file)
    input_len = len(df)
    
    df["main_ec"] = df["labels_str"].apply(extract_main_ec)
    
    # drop identical sequence duplicates (keep first instance)
    df.drop_duplicates(subset=['seq'], keep='first')
    
    
    output_len = len(df)
    
    updated_input_name = input_file.replace(".csv", "")
    
    output_file = f"{input_file}_updated.csv"
    df.to_csv(output_file, index=False)
        
    
    print(f"{input_file} with input length {input_len} updated to {output_file} with lenght {output_len} successfully ... ")
