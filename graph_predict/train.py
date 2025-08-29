# CPU version
import dgl
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    resource_allocation_index,
    preferential_attachment,
    adamic_adar_index
)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_graph(load_path):
    graphs, _ = dgl.load_graphs(load_path)
    return graphs[0]

# Proportion of validation and test sets
test_size = 0.2

# Configuration paths
# graph_path = "/hy-tmp/cot-prompt-words/preprocessing/small_disease.bin"
# model_base_path = "/hy-tmp/cot-prompt-words/rag/graph_model/small"
graph_path = "/hy-tmp/cot-prompt-words/preprocessing/large_disease.bin"
model_base_path = "/hy-tmp/cot-prompt-words/rag/graph_model/large"
os.makedirs(model_base_path, exist_ok=True)

# Load and convert graph
graph = load_graph(graph_path)
multi_g = dgl.to_networkx(graph)
nx_g = nx.Graph()
nx_g.add_nodes_from(multi_g.nodes())
nx_g.add_edges_from(multi_g.edges())

# Construct positive and negative samples
nodes = list(nx_g.nodes())
existing_edges = set(nx_g.edges()) | {(v, u) for u, v in nx_g.edges()}
pos_edges = list(nx_g.edges())
neg_edges = set()
while len(neg_edges) < len(pos_edges):
    u, v = random.sample(nodes, 2)
    if u != v and (u, v) not in existing_edges:
        neg_edges.add((u, v))
neg_edges = list(neg_edges)

all_edges = pos_edges + neg_edges
labels = np.array([1] * len(pos_edges) + [0] * len(neg_edges))

# Calculate link features
def calc_link_features(g, edge_list):
    j = { (u, v): p for u, v, p in jaccard_coefficient(g, edge_list) }
    ra = { (u, v): p for u, v, p in resource_allocation_index(g, edge_list) }
    pa = { (u, v): p for u, v, p in preferential_attachment(g, edge_list) }
    aa = { (u, v): p for u, v, p in adamic_adar_index(g, edge_list) }
    records = []
    for u, v in edge_list:
        records.append({
            'jaccard': j.get((u,v), 0),
            'resource_allocation': ra.get((u,v), 0),
            'preferential_attachment': pa.get((u,v), 0),
            'adamic_adar': aa.get((u,v), 0),
        })
    return pd.DataFrame(records)

edge_df = calc_link_features(nx_g, all_edges)

# Statistical thresholding features
from numpy import percentile

def summarize(arr):
    return np.mean(arr), np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)

# Calculate and save thresholds first
mean_j, med_j, p25_j, p75_j = summarize(edge_df['jaccard'].values)
mean_ra, med_ra, p25_ra, p75_ra = summarize(edge_df['resource_allocation'].values)
mean_pa, med_pa, p25_pa, p75_pa = summarize(edge_df['preferential_attachment'].values)
mean_aa, med_aa, p25_aa, p75_aa = summarize(edge_df['adamic_adar'].values)

import numpy as _np
# Save thresholds
_np.save(
    os.path.join(model_base_path, "thresholds.npy"),
    {
        'jaccard': (mean_j, med_j, p25_j, p75_j),
        'resource_allocation': (mean_ra, med_ra, p25_ra, p75_ra),
        'preferential_attachment': (mean_pa, med_pa, p25_pa, p75_pa),
        'adamic_adar': (mean_aa, med_aa, p25_aa, p75_aa),
    }
)

# Generate binary features based on thresholds
features = {}
for metric, stats in {
    'jaccard': (mean_j, med_j, p25_j, p75_j),
    'resource_allocation': (mean_ra, med_ra, p25_ra, p75_ra),
    'preferential_attachment': (mean_pa, med_pa, p25_pa, p75_pa),
    'adamic_adar': (mean_aa, med_aa, p25_aa, p75_aa)
}.items():
    mean_, median_, p25_, p75_ = stats
    values = edge_df[metric]
    features[f'{metric}_gt_mean'] = (values > mean_).astype(int)
    features[f'{metric}_gt_median'] = (values > median_).astype(int)
    features[f'{metric}_between_25_75'] = values.between(p25_, p75_).astype(int)

feature_df = pd.DataFrame(features)

# Training parameters
EPOCHS = 500
PATIENCE = 20
seeds = list(range(0, 101))

best_seed = None
best_auc = -np.inf
best_model = None

print("=== Multi-seed training and validation ===")
for seed in seeds:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)

    X_train, X_temp, y_train, y_temp = train_test_split(
        feature_df, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    model = SGDClassifier(
        random_state=seed, 
        loss='log_loss', 
        learning_rate='constant', 
        eta0=0.01, 
        warm_start=True
    )
    # Initialize with small samples first
    model.partial_fit(X_train[:10], y_train[:10], classes=[0, 1])

    seed_best_auc = -np.inf
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.partial_fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
        print(f"Epoch {epoch:03d} | Val AUC: {val_auc:.4f}")

        if val_auc > seed_best_auc + 1e-5:
            seed_best_auc = val_auc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Seed {seed} early stopped at Epoch {epoch}")
                break

    # Update global best
    if seed_best_auc > best_auc:
        best_auc = seed_best_auc
        best_seed = seed
        best_model = model

print("\n=== Multi-seed training completed ===")
print(f"Best seed: {best_seed} | Best validation AUC: {best_auc:.4f}")

# Save global best model
final_model_path = os.path.join(model_base_path, f"best_model_seed_{best_seed}.pkl")
joblib.dump(best_model, final_model_path)
print(f"Saved best model to: {final_model_path}")

# Evaluate on test set using best seed
print("\n=== Test set evaluation ===")
set_seed(best_seed)
X_train, X_temp, y_train, y_temp = train_test_split(
    feature_df, labels, test_size=test_size, random_state=best_seed, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=best_seed, stratify=y_temp
)

test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)[:, 1]
from sklearn.metrics import accuracy_score, f1_score

print(f"Test Accuracy : {accuracy_score(y_test, test_preds):.4f}")
print(f"Test F1 Score : {f1_score(y_test, test_preds):.4f}")
print(f"Test ROC AUC  : {roc_auc_score(y_test, test_probs):.4f}")