import os
import networkx as nx
import torch
from torch_geometric.data import Data

# List of GraphML files (update the paths as needed)
folder_path = "graphs/with_features"
graphml_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".graphml")]

def load_graphml(file_path):
    """
    Loads a graph from a GraphML file using NetworkX.
    """
    G = nx.read_graphml(file_path)
    return G

def nx_to_pyg_data(G):
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object.
    Extracts node features based on a defined list of attribute names and 
    uses the "trojan" attribute as the target label.
    """
    # Create a mapping from node names to indices
    node_list = list(G.nodes())
    node_mapping = {node: i for i, node in enumerate(node_list)}

    # Define the full list of feature names (do not include "trojan" here, as it's the label)
    feature_names = [
        #"degree_centrality", "betweenness_centrality", "clustering_coefficient",
        "is_input", "is_output", "is_gate",
        "num_inputs", "num_outputs", 
        "AND", "M", "SDFFSR",
        "AND", "AO21", "AO22", "AOI21", "AOI22", "DFF", "DFFN", "INV", "ISOLAND", "LSDN", "LSDNEN", "MU", "NAND", "NBUFF", "NOR", "OA21", "OA22", "OAI21", "OAI22", "OR", "SDFF",
        "PI", "PO"
    ]

    # Build node feature matrix and label vector
    x = []
    y = []  # Trojan flag label for each node
    for node in node_list:
        feat = []
        for name in feature_names:
            # Convert attribute to float; use 0 if the attribute is missing.
            feat.append(float(G.nodes[node].get(name, 0)))
        x.append(feat)
        # Use the "trojan" attribute as the label (default to 0 if missing)
        y.append(int(G.nodes[node].get("trojan", 0)))
    
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # Define the features you want to normalize
    feature_names_normalization = [
        "num_inputs", "num_outputs", 
        "PI", "PO"
    ]
    # Normalize specific features
    x = normalize_selected_features(x, feature_names, feature_names_normalization)

    # Build edge index from the graph's edges
    edges = []
    for source, target in G.edges():
        src_idx = node_mapping[source]
        tgt_idx = node_mapping[target]
        edges.append([src_idx, tgt_idx])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def normalize_selected_features(x, feature_names, feature_names_normalization):
    """
    Normalizes selected features in the tensor x, based on feature_names_normalization list.
    """
    
    # Find the indices of the features to normalize
    feature_indices = [feature_names.index(f) for f in feature_names_normalization]
    
    # Normalize the selected features
    x_norm = x.clone()
    for idx in feature_indices:
        col = x[:, idx]
        mean = col.mean()
        std = col.std()
        if std > 0:
            x_norm[:, idx] = (col - mean) / std
        else:
            x_norm[:, idx] = col - mean  # Avoid division by zero
    return x_norm

def combine_graphs(graphml_files):
    """
    Combines multiple GraphML files into one large graph.
    It loads each graph using NetworkX, composes them into a single graph,
    and then converts the combined graph into a PyTorch Geometric Data object.
    """
    graphs = []
    for file in graphml_files:
        if os.path.exists(file):
            G = load_graphml(file)
            graphs.append(G)
        else:
            print(f"Warning: File {file} does not exist.")
    
    if not graphs:
        raise ValueError("No graphs loaded. Check your file paths.")
    
    # Compose all graphs into one large graph.
    combined_graph = nx.compose_all(graphs)
    data = nx_to_pyg_data(combined_graph)
    return data

def main():

    # Combine graphs from the provided GraphML files
    combined_data = combine_graphs(graphml_files)
    print("Combined graph data:")
    print(f"  Number of nodes: {combined_data.num_nodes}")
    print(f"  Number of edges: {combined_data.num_edges}")
    print(f"  Node feature shape: {combined_data.x.shape}")
    print(f"  Labels shape: {combined_data.y.shape}")

    # Optionally, save the combined data for later use
    torch.save(combined_data, "combined_graph/combined_graph_data.pt")
    print("Combined graph data saved to combined_graph_data.pt")

if __name__ == "__main__":
    main()
