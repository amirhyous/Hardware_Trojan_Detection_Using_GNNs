import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
import os
from datetime import datetime

# Define the GAT Model
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=6, dropout=0.2):
        super(GATModel, self).__init__()
        # First GAT layer: multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Second GAT layer: projects concatenated features to output classes
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training function (using only training nodes)
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function: compute accuracy over a given mask
def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0
    return accuracy

# Function to plot the confusion matrix (raw counts, not normalized)
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Report additional metrics: Confusion Matrix, TPR, TNR, Precision, Recall, F1 Score, and plot the confusion matrix.
def report_metrics(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    
    true_labels = data.y[mask].cpu().numpy()
    predicted_labels = pred[mask].cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0
        if len(cm) == 1:
            if true_labels[0] == 0:
                TN = cm[0][0]
            else:
                TP = cm[0][0]
    
    # Calculate metrics
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # True Positive Rate (Recall for positive class)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # True Negative Rate (Specificity)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TPR  # Recall is the same as TPR
    f1 = f1_score(true_labels, predicted_labels)
    
    # Plot the confusion matrix (raw counts)
    class_names = ['Non-Trojan', 'Trojan']
    #plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix (Raw Counts)')
    
    metrics = {
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "TPR": TPR,
        "TNR": TNR,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    return metrics

# Function to save results to a CSV file.
def save_results_to_csv(results, file_path="results/results.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as csvfile:
        fieldnames = ["Run", "Timestamp", "TestAccuracy", "TPR", "TNR", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

# Stratified mask creation to preserve class ratios (Trojan nodes ~5%)
def add_masks_stratified(data, train_ratio=0.5, val_ratio=0.1):
    num_nodes = data.num_nodes
    y = data.y

    trojan_idx = (y == 1).nonzero(as_tuple=True)[0]
    non_trojan_idx = (y == 0).nonzero(as_tuple=True)[0]

    trojan_perm = trojan_idx[torch.randperm(trojan_idx.size(0))]
    non_trojan_perm = non_trojan_idx[torch.randperm(non_trojan_idx.size(0))]

    def split_indices(indices, train_ratio, val_ratio):
        n = indices.size(0)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]

    train_t, val_t, test_t = split_indices(trojan_perm, train_ratio, val_ratio)
    train_nt, val_nt, test_nt = split_indices(non_trojan_perm, train_ratio, val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[torch.cat((train_t, train_nt))] = True
    val_mask[torch.cat((val_t, val_nt))] = True
    test_mask[torch.cat((test_t, test_nt))] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def main():
    runs = 1  # Number of times to run the experiment
    for run in range(runs):
        print(f"\n=== Run {run+1}/{runs} ===")
        # Load the combined graph data
        combined_data = torch.load("combined_graph/combined_graph_data.pt")
        
        # If masks do not exist, add them using stratified splitting.
        if not hasattr(combined_data, "train_mask"):
            combined_data = add_masks_stratified(combined_data, train_ratio=0.5, val_ratio=0.1)
        
        print("Graph Data Summary:")
        print(f"  Nodes: {combined_data.num_nodes}")
        print(f"  Edges: {combined_data.num_edges}")
        print(f"  Feature shape: {combined_data.x.shape}")
        print(f"  Label shape: {combined_data.y.shape}")
        print(f"  Train nodes: {combined_data.train_mask.sum().item()}")
        print(f"  Val nodes: {combined_data.val_mask.sum().item()}")
        print(f"  Test nodes: {combined_data.test_mask.sum().item()}")

        # Instantiate the GAT model.
        model = GATModel(
            in_channels=combined_data.num_node_features,
            hidden_channels=128,
            out_channels=2,  # Two classes: Trojan and non-Trojan
            heads=6,
            dropout=0.2
        )

        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        num_epochs = 200
        for epoch in range(num_epochs):
            loss = train(model, combined_data, optimizer, criterion)
            # Optionally, you can print intermediate training/validation accuracies:
            # train_acc = evaluate(model, combined_data, combined_data.train_mask)
            # val_acc = evaluate(model, combined_data, combined_data.val_mask)
            # print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")
            if epoch % 50 == 0:
                torch.save(model.state_dict(), f"model/gat_model_epoch_run{run+1}_{epoch}.pth")

        # After training, evaluate the final model on the test set.
        test_acc = evaluate(model, combined_data, combined_data.test_mask)
        print(f"Run {run+1} - Final Test Accuracy: {test_acc*100:.2f}%")
        
        # Report additional metrics on the test set.
        metrics = report_metrics(model, combined_data, combined_data.test_mask)
        
        # Prepare results dictionary to store in CSV.
        results = {
            "Run": run+1,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "TestAccuracy": test_acc,
            "TPR": metrics["TPR"],
            "TNR": metrics["TNR"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"]
        }
        
        save_results_to_csv(results, file_path="results/results.csv")
        print(f"Run {run+1} results saved to results/results.csv")
        
        torch.save(model.state_dict(), f"model/gat_model_final_run{run+1}.pth")
        print(f"Run {run+1} final model saved to gat_model_final_run{run+1}.pth")

if __name__ == "__main__":
    main()
