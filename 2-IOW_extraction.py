from config import file_name, trojan_gates 
import re
import os
import pandas as pd

def expand_bus_signals(signals):
    expanded_signals = []
    bus_signals = set()  # Store base names of bus signals
    for signal in signals:
        # Check for range notation [msb:lsb]
        match = re.match(r'^(\[\d+:\d+\])\s*(\w+)$', signal)
        if match:
            range_part, base_name = match.groups()
            msb, lsb = map(int, range_part.strip('[]').split(':'))
            if msb >= lsb:
                expanded_signals.extend([f"{base_name}[{i}]" for i in range(lsb, msb + 1)])
            else:
                expanded_signals.extend([f"{base_name}[{i}]" for i in range(msb, lsb + 1)])
            bus_signals.add(base_name)  # Mark this as a bus signal
        else:
            expanded_signals.append(signal.strip())
    return expanded_signals

def merge_IOW(inputs, inputs_completed):
    updated_inputs = inputs.copy()  # Avoid modifying the original inputs list
    
    # For each completed signal, check if it is a base signal for a bus.
    for input_completed in inputs_completed[:]:  # Iterate over a copy of the list
        # If any expanded input starts with input_completed + '[', skip adding the unindexed signal.
        if any(inp.startswith(f"{input_completed}[") for inp in inputs):
            inputs_completed.remove(input_completed)
            continue
        
        # Otherwise, proceed with the original merging logic.
        for input_signal in inputs:
            # Remove the last 3 characters from the signal if it contains '['
            input_tmp = input_signal[:-3] if "[" in input_signal else input_signal
            if input_tmp == input_signal:  # Compare the modified and original signals
                updated_inputs.append(input_completed)
                inputs_completed.remove(input_completed)
                break  # Move to the next input_completed once appended

    return updated_inputs
 
def extract_io_from_verilog(file_path):
    with open(file_path, 'r') as file:
        verilog_code = file.read()

    # Regex patterns for inputs and outputs, capturing the range part and the name
    input_pattern = r'\binput\b(?:\s*(\[\d+:\d+\])\s*|\s+)([a-zA-Z_0-9]+)(?:\s*,\s*|\s*;|\s*)'
    output_pattern = r'\boutput\b(?:\s*(\[\d+:\d+\])\s*|\s+)([a-zA-Z_0-9]+)(?:\s*,\s*|\s*;|\s*)'
    # wire_pattern = r'\bwire\b(?:\s*(\[\d+:\d+\])\s*|\s+)([a-zA-Z_0-9]+)(?:\s*,\s*|\s*;|\s*)'

    # Extract raw inputs and outputs
    raw_inputs = re.findall(input_pattern, verilog_code)
    raw_outputs = re.findall(output_pattern, verilog_code)
    # raw_wires = re.findall(wire_pattern, verilog_code)

    # Combine range and signal name, then split into separate entries
    inputs = [f"{range_part} {name}".strip() for range_part, name in raw_inputs]
    outputs = [f"{range_part} {name}".strip() for range_part, name in raw_outputs]
    # wires = [f"{range_part} {name}".strip() for range_part, name in raw_wires]

    # Expand bus signals
    inputs = expand_bus_signals(inputs)
    outputs = expand_bus_signals(outputs)
    # wires = expand_bus_signals(wires)

    # Regex patterns for inputs and outputs, including comma-separated lists
    input_pattern = r'\binput\b(?:\s*\[.*?\]\s*|\s+)([a-zA-Z_][a-zA-Z_0-9\s,]*)'
    output_pattern = r'\boutput\b(?:\s*\[.*?\]\s*|\s+)([a-zA-Z_][a-zA-Z_0-9\s,]*)'
    # wire_pattern = r'\bwire\b(?:\s*\[.*?\]\s*|\s+)([a-zA-Z_][a-zA-Z_0-9\s,]*)'

    # Extract inputs and outputs
    raw_inputs = re.findall(input_pattern, verilog_code)
    raw_outputs = re.findall(output_pattern, verilog_code)
    # raw_wires = re.findall(wire_pattern, verilog_code)

    # Split comma-separated lists and strip extra spaces
    inputs_completed = [signal.strip() for line in raw_inputs for signal in line.split(',')]
    outputs_completed = [signal.strip() for line in raw_outputs for signal in line.split(',')]
    # wires_completed = [signal.strip() for line in raw_wires for signal in line.split(',')]

    input_final = merge_IOW(inputs, inputs_completed)
    output_final = merge_IOW(outputs, outputs_completed)
    # wire_final = merge_IOW(wires, wires_completed)    

    return input_final, output_final

def save_io_to_csv(inputs, outputs, output_csv_path):
    """
    Saves extracted inputs and outputs to a CSV file.
    """
    # Deduplicate while preserving order
    inputs = list(dict.fromkeys(inputs))
    outputs = list(dict.fromkeys(outputs))
    
    data = []
    for signal in inputs:
        data.append(["input", signal])
    for signal in outputs:
        data.append(["output", signal])

    df = pd.DataFrame(data, columns=["Type", "Signal"])
    df.to_csv(output_csv_path, index=False)
    print(f"Saved inputs and outputs to {output_csv_path}")

# Path to the Verilog file
file_path = f'../dataset/{file_name}.v'

# Extract inputs and outputs
inputs, outputs = extract_io_from_verilog(file_path)

# Save to CSV
output_csv_path = f"netlists/{file_name}_io_signals.csv"
save_io_to_csv(inputs, outputs, output_csv_path)

# Display results
print("Inputs:", inputs)
print("Outputs:", outputs)

