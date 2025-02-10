# Hardware Trojan Detection Using Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

A novel approach for detecting hardware Trojans in integrated circuits using Graph Attention Networks (GATs) at gate-level netlists.

![Workflow Diagram](https://via.placeholder.com/800x400.png?text=Hardware+Trojan+Detection+Workflow)

## 📖 Abstract
With the growing complexity of global semiconductor supply chains, hardware Trojans have become a critical security threat. This project presents a machine learning framework using Graph Neural Networks (GNNs) to detect hardware Trojans in gate-level netlists, achieving:
- **99.32%** overall accuracy
- **88.18%** Trojan detection rate (TPR)
- **93.18%** F1 Score

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/hardware-trojan-detection.git
cd hardware-trojan-detection

# Install dependencies
pip install -r requirements.txt

## 🛠️ Requirements

Python 3.8+
PyTorch 1.10+
PyTorch Geometric
NetworkX
Pandas
Scikit-learn
Matplotlib

## 📂 Project Structure

├── dataset/               # Sample Verilog netlists
├── netlists/             # Processed gate-level information
├── graphs/               # Graph representations
├── results/              # Evaluation metrics
├── model/                # Trained models
│
├── 1-config.py           # Benchmark configurations
├── 2-IOW_extraction.py   # I/O signal extraction
├── 3-netlist_extraction.py # Gate-level extraction
├── 4-netlist_cleaning.py # Netlist preprocessing
├── 5-graph_construction.py # Graph construction
├── 6-feature_extraction.py # GNN feature extraction
├── 7-graph_combination.py # Multi-circuit graph combination
├── 8-GAT_model.py        # GAT training/evaluation
│
└── README.md             # This file

## 🚀 Usage

1. Data Preparation

Place Verilog netlists in dataset/ directory following naming conventions from 1-config.py

2. Run Pipeline

# Step 1-2: Netlist processing
python 2-IOW_extraction.py
python 3-netlist_extraction.py
python 4-netlist_cleaning.py

# Step 3-4: Graph construction
python 5-graph_construction.py
python 6-feature_extraction.py

# Step 5: Combine multiple circuits
python 7-graph_combination.py

# Step 6: Train and evaluate GAT model
python 8-GAT_model.py

## 🎯 Key Features

Automated netlist-to-graph conversion
Structural feature extraction pipeline
Graph Attention Network implementation
Stratified sampling for class imbalance
Comprehensive metrics reporting (TPR, TNR, F1)
Model persistence and results logging

## 📊 Results

Sample output from Trust-HUB benchmarks:

Metric	Value
Accuracy	99.32%
TPR	88.18%
TNR	99.94%
F1 Score	93.18%

## 👨💻 Contributors

Amirhossein Yousefvand (@amirhossein-yousefvand)
Supervisor: Dr. Siamak Mohammadi




