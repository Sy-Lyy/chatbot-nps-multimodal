"""# Sub-RQ3
Which behavioral features contribute most strongly to the multimodal model’s predictive performance, and which contribute the least?

- To measure importance, three complementary analyses are applied to the trained multimodal model:

  - LOFO ablation: each feature is removed (set to zero) and the resulting increase in MAE shows its importance.

  - Permutation importance: feature values are randomly shuffled to test how much prediction accuracy degrades.

  - SHAP analysis: provides feature-wise contribution patterns for individual predictions.

- Together, these methods reveal which behavioral features the model relies on most and least for NPS prediction
"""

# RQ3 — Feature Importance (LOFO, Permutation, SHAP)
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer, AutoModel
import shap
import matplotlib.pyplot as plt


# 1. Setup
device = torch.device("cuda:1")

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Final numeric feature list (NO bot_src)
num_cols = [
    "livechat_flag",
    "activeness_norm",
    "engagement_score_norm",
    "interaction_density_norm",
    "platform_flag",
]

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# 2. Load dataset (same split as training)
df = pd.read_pickle("/home/u807886/Projects/Preprocessed_split7.pkl")
df = df.sort_values("chat_id").reset_index(drop=True)

# Encode chat_id → chat_int
session2idx = {sid: i for i, sid in enumerate(df["chat_id"].unique())}
df["chat_int"] = df["chat_id"].map(session2idx)

# session-level split
sessions = df["chat_id"].unique()

train_sess, test_sess = train_test_split(sessions, test_size=0.2, random_state=42)
train_sess, val_sess  = train_test_split(train_sess, test_size=0.1, random_state=42)

test_df = df[df["chat_id"].isin(test_sess)].reset_index(drop=True)

print("Test sessions:", len(test_sess))
print("Test chunks:", len(test_df))

# 3. Dataset class
class MultiModalChunkDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["text_combined"].tolist()
        self.chatid = df["chat_int"].tolist()
        self.numeric = df[num_cols].values.astype("float32")
        self.labels = df["nps_score"].tolist()

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
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "numeric": torch.tensor(self.numeric[idx], dtype=torch.float32),
            "chat_id": torch.tensor(self.chatid[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# 4. Multimodal model (same as training)
class MultiModalRegressor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("xlm-roberta-base")
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(768 + 32, 1)

    def forward(self, ids, mask, numeric):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0, :]
        num_proj = self.numeric_proj(numeric)
        fused = torch.cat([cls, num_proj], dim=1)
        return self.reg_head(fused).squeeze(-1), cls

# 5. Load trained model
model = MultiModalRegressor(len(num_cols)).to(device)
model.load_state_dict(torch.load("best_multimodal.pt", map_location=device))
model.eval()

print("\nLoaded trained model: best_multimodal.pt\n")

# 6. MAE function
def get_mae(model, df):
    loader = DataLoader(MultiModalChunkDataset(df), batch_size=32)

    session_embs = {}
    session_labels = {}

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            num = batch["numeric"].to(device)
            chat_ids = batch["chat_id"].tolist()
            labels = batch["label"].tolist()

            preds, _ = model(ids, mask, num)
            preds = preds.cpu().numpy()

            for cid, lab, p in zip(chat_ids, labels, preds):
                session_embs.setdefault(cid, []).append(p)
                session_labels[cid] = lab

    session_preds = []
    session_true  = []
    for cid in session_embs:
        session_preds.append(np.mean(session_embs[cid]))
        session_true.append(session_labels[cid])

    return mean_absolute_error(session_true, session_preds)


# 7. LOFO (Leave-One-Feature-Out)
print("========== LOFO ==========\n")

mae_full = get_mae(model, test_df)
print(f"BASELINE MAE = {mae_full:.4f}\n")

lofo_results = []

for col in num_cols:
    df_ab = test_df.copy()
    df_ab[col] = df_ab[col].mean()    # ablate by replacing with global mean
    mae_ab = get_mae(model, df_ab)
    delta = mae_ab - mae_full

    lofo_results.append((col, mae_ab, delta))
    print(f"{col:30s} MAE={mae_ab:.4f} | Δ={delta:.4f}")

print("\nLOFO DONE.\n")

# 8. PERMUTATION IMPORTANCE
print("========== PERMUTATION IMPORTANCE ==========\n")

perm_results = []

for col in num_cols:
    df_perm = test_df.copy()
    df_perm[col] = np.random.permutation(df_perm[col].values)

    mae_perm = get_mae(model, df_perm)
    delta = mae_perm - mae_full
    perm_results.append((col, mae_perm, delta))

    print(f"{col:30s} MAE={mae_perm:.4f} | Δ={delta:.4f}")
print("\nPermutation importance DONE.\n")

# 9. SHAP — numeric-only explanation
print("========== SHAP (numeric features) ==========\n")

# SHAP input
X_num = test_df[num_cols].values.astype("float32")

# ---- Build wrapper that freezes text CLS ----
# Extract frozen CLS (model forward with dummy text)
with torch.no_grad():
    dummy_ids = torch.zeros((1, 512), dtype=torch.long).to(device)
    dummy_mask = torch.zeros((1, 512), dtype=torch.long).to(device)
    cls_frozen = model.encoder(dummy_ids, dummy_mask).last_hidden_state[:, 0, :].squeeze()

class NumericOnly(nn.Module):
    def __init__(self, multimodal, cls_vec):
        super().__init__()
        self.numeric_proj = multimodal.numeric_proj
        self.reg_head = multimodal.reg_head
        self.cls = cls_vec  # fxed CLS embedding

    def forward(self, x):
        num_p = self.numeric_proj(x)
        fused = torch.cat([self.cls.expand(len(x), -1), num_p], dim=1)
        return self.reg_head(fused).squeeze(-1)

numeric_model = NumericOnly(model, cls_frozen).to(device)

# SHAP wrapper
def predict_numeric(x):
    x = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = numeric_model(x).cpu().numpy()
    return out

# background
background = shap.sample(X_num, 100)

explainer = shap.KernelExplainer(predict_numeric, background)

test_sample = shap.sample(X_num, 200)
shap_values = explainer.shap_values(test_sample)

shap.summary_plot(shap_values, test_sample, feature_names=num_cols)
shap.summary_plot(shap_values, test_sample, feature_names=num_cols, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")

print("\nSHAP DONE.\n")
