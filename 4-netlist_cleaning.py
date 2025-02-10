import pandas as pd
import os
from config import file_name, trojan_gates 

def clean_signal_name(signal):
    """
    Replaces '\\' and '/' in signal names with '_'.
    Also replaces specific signal names as required.
    """
    if isinstance(signal, str):  # Ensure it's a string
        signal = signal.replace("\\", "_").replace("/", "_")  # Replace slashes
        signal = signal.replace("_test_point_TM", "test_mode")  # Replace specific name
        signal = signal.replace("rec_dataH_temp[7]", "test_so")  # Replace specific name
    return signal  # Return cleaned signal

def get_gate_family(gate_type):
    if gate_type.startswith("AOI") or gate_type.startswith("OAI"):
        return gate_type[:5]
    elif gate_type.startswith("AO") or gate_type.startswith("OA"):
        return gate_type[:4]
    else:
        return ''.join([char for char in gate_type if not char.isdigit() and char != 'X'])

def process_csv(csv_path, output_path):

    file_name = ((os.path.splitext(os.path.basename(file_path))[0]).split("_"))[0]
    print(file_name)
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Apply the cleaning function to Inputs and Outputs columns
    df["Inputs"] = df["Inputs"].dropna().apply(lambda x: ", ".join(clean_signal_name(sig.strip()) for sig in str(x).split(", ")))
    df["Outputs"] = df["Outputs"].dropna().apply(lambda x: ", ".join(clean_signal_name(sig.strip()) for sig in str(x).split(", ")))
    
    # Create the Gate Family column
    df["Gate Family"] = df["Gate Type"].apply(get_gate_family)
    
    # Reorder columns to place Gate Family after Gate Type
    columns = df.columns.tolist()
    gate_type_index = columns.index("Gate Type")
    # Remove "Gate Family" and insert it right after "Gate Type"
    columns.insert(gate_type_index + 1, columns.pop(columns.index("Gate Family")))
    df = df[columns]
    
    # Create the Full Gate Name column: concatenate file_name and Gate Name
    df["Full Gate Name"] = file_name + "_" + df["Gate Name"]
    
    # Reorder columns to place Full Gate Name immediately after Gate Name
    columns = df.columns.tolist()
    gate_name_index = columns.index("Gate Name")
    # Remove "Full Gate Name" and insert it right after "Gate Name"
    columns.insert(gate_name_index + 1, columns.pop(columns.index("Full Gate Name")))
    df = df[columns]
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

# Example usage

file_path = f'netlists/{file_name}_gates_info.csv'

output_path = f"netlists/{file_name}_gates_info_cleaned.csv"

process_csv(file_path, output_path)
