import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from config import file_name, trojan_gates_full 

def construct_graph_with_internal_wiring(gate_csv_path, io_csv_path, trojan_gates_full):
    # 1. Load I/O signals
    io_df = pd.read_csv(io_csv_path)
    
    # Dictionary to store I/O signal types
    port_nodes = {}
    for _, row in io_df.iterrows():
        sig = row["Signal"].strip()
        typ = row["Type"].strip().lower()
        if typ == "input":
            port_nodes[sig] = "input_port"
        elif typ == "output":
            port_nodes[sig] = "output_port"
    
    # 2. Load gate information
    gate_df = pd.read_csv(gate_csv_path)
    
    # 3. Initialize directed graph
    G = nx.DiGraph()

    # 4. Add port nodes (from I/O signals)
    for sig, port_type in port_nodes.items():
        G.add_node(sig, node_type="port", gate_type=port_type)
    
    # 5. Add gate nodes
    output_mapping = {}  # Map each signal to the gate(s) that produce it
    for _, row in gate_df.iterrows():
        gate_name = row["Full Gate Name"].strip()
        gate_type = row["Gate Family"].strip() if pd.notna(row["Gate Family"]) else "unknown"
        is_trojan = 1 if gate_name in trojan_gates_full else 0
        
        G.add_node(gate_name, node_type="gate", gate_type=gate_type, trojan=is_trojan)
        
        # Process outputs and store them in the output_mapping dictionary
        if pd.notna(row["Outputs"]):
            outputs = [sig.strip() for sig in str(row["Outputs"]).split(",")]
        else:
            outputs = []
        
        for out_signal in outputs:
            if out_signal in ["1'b1", "1'b0"]:
                continue
            if out_signal not in output_mapping:
                output_mapping[out_signal] = []
            output_mapping[out_signal].append(gate_name)
    
    # 6. Connect input ports to gates and gates to each other
    for _, row in gate_df.iterrows():
        gate_name = row["Full Gate Name"].strip()
        
        # Process inputs
        if pd.notna(row["Inputs"]):
            inputs = [sig.strip() for sig in str(row["Inputs"]).split(",")]
        else:
            inputs = []
        
        for in_signal in inputs:
            if in_signal in ["1'b1", "1'b0"]:
                continue
            
            if in_signal in port_nodes:  # Input comes from an external port
                G.add_edge(in_signal, gate_name)
            elif in_signal in output_mapping:  # Input comes from another gate's output
                for source_gate in output_mapping[in_signal]:
                    if source_gate != gate_name:  # Avoid self-loops
                        G.add_edge(source_gate, gate_name)
    
    # 7. Connect gates to output ports
    for _, row in gate_df.iterrows():
        gate_name = row["Full Gate Name"].strip()
        
        if pd.notna(row["Outputs"]):
            outputs = [sig.strip() for sig in str(row["Outputs"]).split(",")]
        else:
            outputs = []
        
        for out_signal in outputs:
            if out_signal in ["1'b1", "1'b0"]:
                continue
            
            if out_signal in port_nodes:  # Output goes to an external port
                G.add_edge(gate_name, out_signal)
    
    return G


if __name__ == "__main__":
    # Define paths
    gate_csv_path = f"netlists/{file_name}_gates_info_cleaned.csv"

    io_csv_path = f"netlists/{file_name}_io_signals.csv"

    # Construct graph
    circuit_graph = construct_graph_with_internal_wiring(gate_csv_path, io_csv_path, trojan_gates_full)

    # Save the complete graph to a GraphML file (you can also choose GEXF or JSON)
    nx.write_graphml(circuit_graph, f"graphs/{file_name}_graph.graphml")
    print(f"Graph saved to features/{file_name}_graph.graphml")

    # (Optional) You can still extract features if needed for your GNN\n       
    # feature_df = extract_features(circuit_graph, gate_families)     
    # feature_df.to_csv("features/gnn_features.csv", index=False)  
    # print("Feature extraction complete. Saved to features/gnn_features.csv")








# # Paths to CSV files
# gate_csv_path = "netlist/gates_info_cleaned.csv"
# io_csv_path = "netlist/io_signals.csv"

# # Construct the graph
# circuit_graph = construct_graph_with_internal_wiring(gate_csv_path, io_csv_path)

# # Print graph information
# print("Total nodes:", circuit_graph.number_of_nodes())
# print("Total edges:", circuit_graph.number_of_edges())

# # Inspect a few nodes
# for node, data in list(circuit_graph.nodes(data=True))[:10]:
#     print(f"Node: {node}, Attributes: {data}")

# # Visualization
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(circuit_graph, k=0.15, iterations=20)

# # Color nodes based on type
# node_colors = []
# for node, data in circuit_graph.nodes(data=True):
#     if data.get("node_type") == "gate":
#         node_colors.append("skyblue")
#     elif data.get("node_type") == "port":
#         node_colors.append("lightgreen" if data.get("gate_type") == "input_port" else "orange")

# nx.draw(circuit_graph, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, edge_color='gray')
# plt.title("Full Circuit Graph with Internal Wiring")
# plt.show()


# # Highlight specific gates (e.g., Trojan gates)
# trojan_nodes = [node for node, data in circuit_graph.nodes(data=True) if data.get("trojan", 0) == 1]

# # Create a subgraph of Trojan nodes
# subgraph = circuit_graph.subgraph(trojan_nodes)

# # Plot the subgraph
# pos = nx.spring_layout(subgraph)
# plt.figure(figsize=(10, 8))
# nx.draw(subgraph, pos, with_labels=True, node_size=1000, font_size=8, edge_color='red')
# plt.title("Trojan Node Subgraph")
# plt.show()