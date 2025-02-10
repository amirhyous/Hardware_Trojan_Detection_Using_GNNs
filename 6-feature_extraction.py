import networkx as nx
import numpy as np
import pandas as pd
from graph_construction import construct_graph_with_internal_wiring
from config import file_name, trojan_gates_full 

def extract_gate_families(gate_csv_path):
    """
    Extracts unique gate families from the CSV file.
    """
    df = pd.read_csv(gate_csv_path)
    if "Gate Family" in df.columns:
        return sorted(df["Gate Family"].dropna().unique())
    return []

def extract_features(graph, gate_families):
    """
    Extracts node features for GNN input.
    
    Features:
    - Degree centrality
    - Betweenness centrality
    - Clustering coefficient
    - Number of inputs (incoming edges)
    - Number of outputs (outgoing edges)
    - Trojan flag (if applicable)
    - One-hot encoding of node type (input, output, gate)
    - One-hot encoding of gate family
    - PI: shortest path length (undirected) to the nearest input port
    - PO: shortest path length (undirected) to the nearest output port
    
    Returns:
    - A DataFrame where each row corresponds to a node with its features.
    """
    # Compute centrality metrics
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
    clustering_coefficient = nx.clustering(graph)

    # --- Compute PI and PO for each node ---
    # Convert the graph to undirected to compute shortest path distances
    G_undirected = graph.to_undirected()

    # Identify input and output port nodes based on their attributes.
    input_nodes = [n for n, d in graph.nodes(data=True)
                   if d.get("node_type") == "port" and d.get("gate_type") == "input_port"]
    output_nodes = [n for n, d in graph.nodes(data=True)
                    if d.get("node_type") == "port" and d.get("gate_type") == "output_port"]

    # Use multi_source_dijkstra_path_length as an alternative to multi_source_shortest_path_length.
    pi_all = nx.multi_source_dijkstra_path_length(G_undirected, input_nodes) if input_nodes else {}
    po_all = nx.multi_source_dijkstra_path_length(G_undirected, output_nodes) if output_nodes else {}

    # Extract node features
    node_features = []
    for node, data in graph.nodes(data=True):
        # Retrieve attributes
        node_type = data.get("node_type", "unknown")
        # For gate nodes, gate_type stores the gate family; for port nodes, it holds "input_port" or "output_port"
        gate_family = data.get("gate_type", "unknown")
        trojan = data.get("trojan", 0)

        # Compute number of inputs (incoming edges) and number of outputs (outgoing edges)
        num_inputs = graph.in_degree(node) if graph.is_directed() else sum(1 for _ in graph.predecessors(node))
        num_outputs = graph.out_degree(node) if graph.is_directed() else sum(1 for _ in graph.successors(node))

        # One-hot encoding for node type:
        if node_type == "port":
            is_input = 1 if gate_family == "input_port" else 0
            is_output = 1 if gate_family == "output_port" else 0
            is_gate = 0
        elif node_type == "gate":
            is_input = 0
            is_output = 0
            is_gate = 1
        else:
            is_input = 0
            is_output = 0
            is_gate = 0
        node_type_encoding = [is_input, is_output, is_gate]

        # One-hot encoding for gate family (for gate nodes only; port nodes will have all zeros)
        gate_family_encoding = [1 if gate_family == gf else 0 for gf in gate_families]

        # Compute PI and PO (shortest path length to nearest input and output port respectively)
        # If the node itself is an input or output port, set the corresponding value to 0.
        if node_type == "port" and gate_family == "input_port":
            PI = 0
        else:
            PI = pi_all.get(node, float('inf'))

        if node_type == "port" and gate_family == "output_port":
            PO = 0
        else:
            PO = po_all.get(node, float('inf'))

        # Feature vector
        feature_vector = [
            degree_centrality.get(node, 0),
            betweenness_centrality.get(node, 0),
            clustering_coefficient.get(node, 0),
            num_inputs,
            num_outputs,
            trojan
        ] + node_type_encoding + gate_family_encoding + [PI, PO]

        node_features.append((node, feature_vector))
    
    # Define feature column names (appending PI and PO at the end)
    feature_columns = (
        ["degree_centrality", "betweenness_centrality", "clustering_coefficient", "num_inputs", "num_outputs", "trojan"] +
        ["is_input", "is_output", "is_gate"] +
        gate_families +
        ["PI", "PO"]
    )
    feature_df = pd.DataFrame(node_features, columns=["Node", "Features"])
    feature_df[feature_columns] = pd.DataFrame(feature_df["Features"].tolist(), index=feature_df.index)
    feature_df.drop(columns=["Features"], inplace=True)
    
    return feature_df

if __name__ == "__main__":
    # Define file paths
    gate_csv_path = f"netlists/{file_name}_gates_info_cleaned.csv"
    io_csv_path = f"netlists/{file_name}_io_signals.csv"

    # Extract gate families from the CSV file
    gate_families = extract_gate_families(gate_csv_path)
    print("Gate families:", gate_families)

    # Construct the graph
    circuit_graph = construct_graph_with_internal_wiring(gate_csv_path, io_csv_path, trojan_gates_full)

    # Extract node features
    feature_df = extract_features(circuit_graph, gate_families)
    feature_df.to_csv('features.csv', index=False)

    # Update each node's attributes in the graph with its computed features
    for _, row in feature_df.iterrows():
        node = row["Node"]
        for col in feature_df.columns:
            if col != "Node":
                circuit_graph.nodes[node][col] = row[col]

    # Save the complete graph (with connectivity, directions, and features) to a GraphML file
    nx.write_graphml(circuit_graph, f"graphs/{file_name}_graph_with_features.graphml")
    print(f"Graph saved to graphs/{file_name}_graph_with_features.graphml")
