import re
import os
import pandas as pd
from config import file_name, trojan_gates 

def extract_gates_from_verilog(file_path):

    with open(file_path, 'r') as file:
        verilog_code = file.read()

    # Regex pattern to match gate declarations spanning multiple lines
    gate_pattern = re.compile(
        r'\b(?P<gate_type>[a-zA-Z_0-9]+)\s+(?P<gate_name>[a-zA-Z_0-9]+)\s*\((?P<connections>.*?)\);',
        re.DOTALL  # Allows matching across multiple lines
    )

    # List to store gate data
    gates_data = []

    # Modules to ignore
    ignored_modules = ["uart"]

    # Match and extract gate details
    for match in gate_pattern.finditer(verilog_code):
        gate_type = match.group('gate_type')
        gate_name = match.group('gate_name')
        connections = match.group('connections')

        # Skip ignored modules
        if gate_type.lower() in ignored_modules:
            continue

        # Flatten the connections block while preserving signals split across lines
        connections = re.sub(r'\s*\n\s*', '', connections)

        # It detects 1'b1 and Slashes:
        connection_pattern = re.compile(
            r'\.(?P<port>[A-Za-z_0-9]+)\(\s*(?P<signal>\\?[A-Za-z_0-9\[\]/]+|[01]\'b[01])\s*\)'
        )

        # Updated regex to match special characters in signal names, including constants like 1'b1 or 1'b0
        # connection_pattern = re.compile(
        #     r'\.(?P<port>[A-Za-z_0-9]+)\((?P<signal>\\?[A-Za-z_0-9\[\]/]+|[01]\'b[01])\)'
        # )

        # Does not detect 1'b1:
        # connection_pattern = re.compile(
        #     r'\.(?P<port>[A-Za-z_0-9]+)\((?P<signal>\\?[A-Za-z_0-9\[\]/]+)\)'
        # )

        ports = connection_pattern.findall(connections)

        inputs = []
        outputs = []
        for port, signal in ports:
            # Assume ports named "Y", "ZN", "OUT", "Q", "QN", or those starting with "Z" are outputs
            if port in ['Y', 'ZN', 'OUT', 'Q', 'QN'] or port.startswith('Z'):
                outputs.append(signal)
            else:
                inputs.append(signal)

        # Skip gates without valid inputs or outputs
        if not inputs and not outputs:
            continue

        # Append data as a dictionary to the list
        gates_data.append({
            'Gate Type': gate_type,
            'Gate Name': gate_name,
            'Inputs': ', '.join(inputs),
            'Outputs': ', '.join(outputs)
        })

    # Convert list of dictionaries to a DataFrame
    gates_df = pd.DataFrame(gates_data)
    return gates_df


# Path to the Verilog file
file_path = f'../dataset/{file_name}.v'

# Extract gates and store in a DataFrame
gates_df = extract_gates_from_verilog(file_path)

# List of gates to mark as Trojan

# Add a Trojan column and set value to 1 for the specified gates, otherwise 0
gates_df['Trojan'] = gates_df['Gate Name'].apply(lambda name: 1 if name in trojan_gates else 0)

# Save the DataFrame to a CSV file (optional)
gates_df.to_csv(f"netlists/{file_name}_gates_info.csv", index=False)

# Display the DataFrame
print(gates_df)

