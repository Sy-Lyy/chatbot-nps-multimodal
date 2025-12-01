"""# Sub-RQ2
How does the performance gain of multimodal models vary across subgroups?

- Subgroup definitions
  1. Activeness — low / mid / high
  2. Engagement Score — low / mid / high
  3. Interaction Density — low / mid / high

   
  - *Low, mid, high groups were formed using tertile (33% quantile) binning to ensure balanced subgroup sizes.*



- Excluded subgroup
  - Platform flag and Livechat flag were removed due to severe class imbalance.

- Evaluation procedure
For each of the 10 subgroups, the best unimodal and multimodal models were evaluated on the test set using MAE to quantify subgroup-specific performance differences.

"""

# RQ2 — Subgroup performance comparison
# Δ = MAE_uni - MAE_multi

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# 1. Reproducibility Setup
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:1")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

num_cols = [
    "livechat_flag",
    "activeness_norm",
    "engagement_score_norm",
    "interaction_density_norm",
    "platform_flag"
]

# 2. Load Same Dataset / Same Split as Training
df = pd.read_pickle("/home/u807886/Projects/Preprocessed_split7.pkl")
df = df.sort_values("chat_id").reset_index(drop=True)

# Encode chat_id (string → integer)
session2idx = {sid: i for i, sid in enumerate(df["chat_id"].unique())}
df["chat_int"] = df["chat_id"].map(session2idx)

# session-level split (same logic used during training)
sessions = df["chat_id"].unique()

train_sess, test_sess = train_test_split(sessions, test_size=0.2, random_state=42)
train_sess, val_sess = train_test_split(train_sess, test_size=0.1, random_state=42)

test_df = df[df["chat_id"].isin(test_sess)].reset_index(drop=True)


# 3. Model Definitions
class TextRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("xlm-roberta-base")
        self.reg_head = nn.Linear(768, 1)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state[:, 0, :]  # chunk CLS


class MultiModalRegressor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("xlm-roberta-base")
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.reg_head = nn.Linear(768 + 32, 1)

    def forward(self, ids, mask, numeric):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        text_emb = out.last_hidden_state[:, 0, :]
        num_emb = self.numeric_proj(numeric)
        return torch.cat([text_emb, num_emb], dim=1)


# 4. Load trained weights
uni_model = TextRegressor().to(device)
uni_model.load_state_dict(torch.load("best_unimodal.pt", map_location=device))
uni_model.eval()

multi_model = MultiModalRegressor(len(num_cols)).to(device)
multi_model.load_state_dict(torch.load("best_multimodal.pt", map_location=device))
multi_model.eval()

# 5. Datasets
class ChunkDataset(Dataset):
    def __init__(self, df, numeric=None):
        self.texts = df["text_combined"].tolist()
        self.chat_ids = df["chat_int"].tolist()
        self.labels = df["nps_score"].tolist()
        self.numeric = df[numeric].values.astype("float32") if numeric else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "chat_id": torch.tensor(self.chat_ids[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        if self.numeric is not None:
            item["numeric"] = torch.tensor(self.numeric[idx], dtype=torch.float32)

        return item

# 6. Prediction functions
def predict_unimodal(df):
    loader = DataLoader(ChunkDataset(df), batch_size=32)

    session_embs = {}
    session_labels = {}

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            chat_ids = batch["chat_id"].tolist()
            labels = batch["label"].tolist()

            chunk_emb = uni_model(ids, mask).cpu().numpy()

            for cid, lab, emb in zip(chat_ids, labels, chunk_emb):
                session_embs.setdefault(cid, []).append(emb)
                session_labels[cid] = lab

    preds, labels = [], []
    for cid in session_embs:
        pooled = np.mean(session_embs[cid], axis=0)
        pooled = torch.tensor(pooled, dtype=torch.float32).to(device).unsqueeze(0)

        pred = uni_model.reg_head(pooled).cpu().item()

        preds.append(pred)
        labels.append(session_labels[cid])

    return np.array(preds), np.array(labels)



def predict_multimodal(df):
    loader = DataLoader(ChunkDataset(df, numeric=num_cols), batch_size=32)

    session_embs = {}
    session_labels = {}

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            num = batch["numeric"].to(device)
            chat_ids = batch["chat_id"].tolist()
            labels = batch["label"].tolist()

            chunk_emb = multi_model(ids, mask, num).cpu().numpy()

            for cid, lab, emb in zip(chat_ids, labels, chunk_emb):
                session_embs.setdefault(cid, []).append(emb)
                session_labels[cid] = lab

    preds, labels = [], []
    for cid in session_embs:
        pooled = np.mean(session_embs[cid], axis=0)
        pooled = torch.tensor(pooled, dtype=torch.float32).to(device).unsqueeze(0)

        pred = multi_model.reg_head(pooled).cpu().item()

        preds.append(pred)
        labels.append(session_labels[cid])

    return np.array(preds), np.array(labels)

# 7. Subgroup Binning
test_df["act_group"] = pd.qcut(test_df["activeness_norm"], 3, labels=["low", "mid", "high"])
test_df["eng_group"] = pd.qcut(test_df["engagement_score_norm"], 3, labels=["low", "mid", "high"])
test_df["dens_group"] = pd.qcut(test_df["interaction_density_norm"], 3, labels=["low", "mid", "high"])


# 8. Subgroup evaluation + bootstrap CI
def subgroup_results(df, col):
    results = []
    for g in sorted(df[col].dropna().unique()):
        g_df = df[df[col] == g]

        up, y = predict_unimodal(g_df)
        mp, _ = predict_multimodal(g_df)

        delta = mean_absolute_error(y, up) - mean_absolute_error(y, mp)

        # bootstrap CI
        boot = []
        for _ in range(300):
            idx = np.random.choice(len(y), len(y), replace=True)
            boot.append(
                mean_absolute_error(y[idx], up[idx]) -
                mean_absolute_error(y[idx], mp[idx])
            )
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

        results.append((g, delta, ci_low, ci_high))
    return results

# 9. Print results
print("\n======== Activeness Subgroups ========")
for g, d, c1, c2 in subgroup_results(test_df, "act_group"):
    print(f"{g}: Δ={d:.4f} (CI [{c1:.4f}, {c2:.4f}])")

print("\n======== Engagement Subgroups ========")
for g, d, c1, c2 in subgroup_results(test_df, "eng_group"):
    print(f"{g}: Δ={d:.4f} (CI [{c1:.4f}, {c2:.4f}])")

print("\n======== Interaction Density Subgroups ========")
for g, d, c1, c2 in subgroup_results(test_df, "dens_group"):
    print(f"{g}: Δ={d:.4f} (CI [{c1:.4f}, {c2:.4f}])")

print("\nDone.")
