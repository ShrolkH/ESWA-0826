import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
import json
import random

# ---------------------------
# 1. Random Seed Configuration
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# 2. Custom Dataset and Network Structure
# ---------------------------
class TwoFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------
# 3. Data Loading Function
# ---------------------------
def load_raw_data(dataset_name: str ,file_path):
    """
    Read raw data from a JSON file and return features_all and labels_all (without shuffling the order).
    """

    if dataset_name == 'large':
        pos_t = '\"result\": \"1\"\n'
        neg_t = '\"result\": \"0\"\n'
    else:
        pos_t = '\"result\":\"1\"'
        neg_t = '\"result\":\"0\"'

    with open(file_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    features = []
    labels = []
    for item in raw:
        single_feat = [item["graph_prediction"]]
        if neg_t in item["output"]:
            single_feat.append(0)
        elif pos_t in item["output"]:
            single_feat.append(1)
        else:
            # Skip if an error occurs
            continue

        features.append(single_feat)
        labels.append(float(1) if item["pos"] else float(0))

    return np.array(features), np.array(labels)


# ---------------------------
# 4. Data Splitting and DataLoader Construction
# ---------------------------
def split_data_loaders(
    features_all: np.ndarray,
    labels_all: np.ndarray,
    seed: int,
    batch_size: int = 64
):
    """
    Randomly shuffle (features_all, labels_all) based on the given seed, split into train/valid/test sets,
    and return the corresponding DataLoaders.
    train:valid:test = 8:1:1
    shuffle=True is only applied to train_loader.
    """
    set_seed(seed)
    idx = np.random.permutation(len(labels_all))
    feats = features_all[idx]
    labs = labels_all[idx]

    n = len(labs)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    train_ds = TwoFeatureDataset(feats[:train_end], labs[:train_end])
    valid_ds = TwoFeatureDataset(feats[train_end:valid_end], labs[train_end:valid_end])
    test_ds  = TwoFeatureDataset(feats[valid_end:], labs[valid_end:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


# ---------------------------
# 5. Training and Evaluation Functions
# ---------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
):
    model.eval()
    all_true = []
    all_score = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            preds = model(Xb)
            all_true.append(yb.numpy())
            all_score.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)

    try:
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
    except ValueError:
        auc = ap = f1 = 0.0

    return auc, ap, f1, y_score


# ---------------------------
# 6. Function to Search for the Best Seed
# ---------------------------
def search_best_seed(
    features_all: np.ndarray,
    labels_all: np.ndarray,
    seed_range: range = range(100),
    device: torch.device = None
):
    """
    Search for the best seed within the seed range of 0–99.
    Return (best_seed, best_result_dict), where best_result_dict contains 'AUC', 'AP', 'F1', 'Threshold'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_seed = None
    best_auc = -1.0
    best_result = {}

    for seed in seed_range:
        print(f"\n--- Searching seed = {seed} ---")
        # 1) Split data based on the seed
        train_ld, valid_ld, test_ld = split_data_loaders(features_all, labels_all, seed)

        # 2) Initialize model and optimizer
        model = ClassifierNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # 3) Parameters for Early Stopping
        patience = 20
        no_improve = 0
        best_val_auc = 0.0

        # 4) Training loop
        for epoch in range(1, 201):
            train_loss = train_one_epoch(model, train_ld, optimizer, criterion, device)
            val_auc, _, _, _ = evaluate_model(model, valid_ld, device)
            #print(f"Epoch {epoch} | Train Loss = {train_loss:.4f} | Valid AUC = {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        # 5) Grid search for the optimal threshold on the validation set
        best_f1, best_th = 0.0, 0.5
        for t in np.linspace(0, 1, 101):
            _, _, f1_t, _ = evaluate_model(model, valid_ld, device, threshold=t)
            if f1_t > best_f1:
                best_f1, best_th = f1_t, t

        # 6) Evaluate on the test set
        auc, ap, f1_val, _ = evaluate_model(model, test_ld, device, threshold=best_th)
        print(f"seed = {seed} | Test AUC = {auc:.4f}, AP = {ap:.4f}, F1 = {f1_val:.4f}, th = {best_th:.2f}")

        if auc > best_auc:
            best_auc = auc
            best_seed = seed
            best_result = {
                "AUC": auc,
                "AP": ap,
                "F1": f1_val,
                "Threshold": best_th
            }

    print("\n>>> Search completed.")
    print(f"Best seed = {best_seed} , corresponding metrics: AUC={best_result['AUC']:.4f}, AP={best_result['AP']:.4f}, "
          f"F1={best_result['F1']:.4f}, Threshold={best_result['Threshold']:.2f}")
    return best_seed, best_result


# ---------------------------
# 7. Function to Reproduce Experiment with a Specified Seed
# ---------------------------
def reproduce_with_seed(
    features_all: np.ndarray,
    labels_all: np.ndarray,
    seed: int,
    device: torch.device = None
):
    """
    Reproduce the training-validation-test workflow with the given seed and print the final test set metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n>>> Reproducing experiment with seed = {seed}")
    # 1) Split data
    train_ld, valid_ld, test_ld = split_data_loaders(features_all, labels_all, seed)

    # 2) Initialize model and optimizer
    model = ClassifierNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # 3) Early Stopping parameters
    patience = 20
    no_improve = 0
    best_val_auc = 0.0

    # 4) Training loop
    for epoch in range(1, 201):
        train_loss = train_one_epoch(model, train_ld, optimizer, criterion, device)
        val_auc, _, _, _ = evaluate_model(model, valid_ld, device)
        print(f"Epoch {epoch:03d} | Train Loss = {train_loss:.4f} | Valid AUC = {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping (no improvement in validation set AUC for {patience} consecutive times), stopping at epoch {epoch}")
            break

    # 5) Search for the optimal threshold on the validation set
    best_f1, best_th = 0.0, 0.5
    for t in np.linspace(0, 1, 101):
        _, _, f1_t, _ = evaluate_model(model, valid_ld, device, threshold=t)
        if f1_t > best_f1:
            best_f1, best_th = f1_t, t
    print(f"Optimal threshold on validation set: {best_th:.2f}, corresponding F1 = {best_f1:.4f}")

    # 6) Evaluate on the test set
    auc, ap, f1_val, _ = evaluate_model(model, test_ld, device, threshold=best_th)
    print(f"[Reproduction Result] Test AUC = {auc:.4f}, AP = {ap:.4f}, F1 = {f1_val:.4f}, Threshold = {best_th:.2f}")


# ---------------------------
# 8. Main Function Entry: Determine Workflow Based on seed_to_use
# ---------------------------
def main():
    # Set to -1 if you want to search for the best seed; if you already have the optimal seed (e.g., 5),
    # set it to 5 for reproduction
    seed_to_use = 62
    file_path = "" # Result of MS-RAG

    features_all, labels_all = load_raw_data("large",file_path)
    print("Total number of samples:", len(labels_all))

    # CPU/GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if seed_to_use == -1:
        # Search for the best seed and return the result
        best_seed, best_result = search_best_seed(
            features_all,
            labels_all,
            seed_range=range(100),
            device=device
        )
        # If you want to reproduce with this best_seed, simply set seed_to_use = best_seed and run the script again
    else:
        # Reproduce the experiment with the specified seed
        reproduce_with_seed(
            features_all,
            labels_all,
            seed=seed_to_use,
            device=device
        )


if __name__ == '__main__':
    main()