# Main RQ
How effectively does a transformer-based multimodal model incorporating text and behavioral features outperform a unimodal transformer text model in predicting NPS?

# Sub-RQ1
How much predictive gain is achieved by augmenting an XLM-RoBERTa text model with behavioral features in a multimodal architecture for NPS prediction?

- Test MAE is compared among Baseline, Unimodal, and Multimodal models.
"""

# Error Analysis
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")  # 터미널/서버 환경에서 그림 저장용 backend
import matplotlib.pyplot as plt

# ============================================================
# 0. Output dir for figures
# ============================================================
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. Reproducibility & Device
# ============================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:1")

# ============================================================
# 2. Load dataset & session-based split (same logic as training)
# ============================================================
df = pd.read_pickle("/home/u807886/Projects/Preprocessed_split7.pkl")
df = df.sort_values("chat_id").reset_index(drop=True)

# chat_id → int
session2idx = {sid: i for i, sid in enumerate(df["chat_id"].unique())}
df["chat_int"] = df["chat_id"].map(session2idx)

sessions = df["chat_id"].unique()
train_sess, test_sess = train_test_split(sessions, test_size=0.2, random_state=42)
train_sess, val_sess  = train_test_split(train_sess, test_size=0.1, random_state=42)

test_df = df[df["chat_id"].isin(test_sess)].reset_index(drop=True)
print(f"Test chunks: {len(test_df)}")

# ============================================================
# 3. Features & tokenizer
# ============================================================
num_cols = [
    "livechat_flag",
    "activeness_norm",
    "engagement_score_norm",
    "interaction_density_norm",
    "platform_flag",
]

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_texts(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

# ============================================================
# 4. Session-level dataset (1 session = 1 item)
# ============================================================
class SessionDataset(Dataset):
    def __init__(self, df, num_cols):
        self.sessions = df["chat_id"].unique()
        self.df = df
        self.num_cols = num_cols

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sid = self.sessions[idx]
        sess = self.df[self.df["chat_id"] == sid]

        texts = sess["text_combined"].tolist()
        enc = tokenize_texts(texts)  # input_ids: (C, 512)

        numeric = sess[self.num_cols].iloc[0].values.astype(np.float32)
        label = sess["nps_score"].iloc[0]

        return {
            "input_ids": enc["input_ids"],            # (C, 512)
            "attention_mask": enc["attention_mask"],  # (C, 512)
            "numeric": torch.tensor(numeric, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }

test_loader = DataLoader(SessionDataset(test_df, num_cols), batch_size=1, shuffle=False)

# ============================================================
# 5. Model definitions (same as training architecture)
# ============================================================
class TextRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("xlm-roberta-base")
        self.reg_head = nn.Linear(768, 1)

    def forward(self, ids, mask):
        # ids: (C, L)
        C, L = ids.shape
        out = self.encoder(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0, :]  # (C, 768)
        session_emb = cls.mean(dim=0)         # (768,)
        pred = self.reg_head(session_emb.unsqueeze(0))  # (1, 1)
        return pred.squeeze(-1)               # scalar


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
        C, L = ids.shape
        out = self.encoder(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0, :]  # (C, 768)
        session_emb = cls.mean(dim=0)         # (768,)

        numeric = numeric.view(-1)            # (num_features,)
        num_emb = self.numeric_proj(numeric.unsqueeze(0))  # (1, 32)

        fused = torch.cat([session_emb, num_emb.squeeze(0)], dim=0)  # (800,)
        pred = self.reg_head(fused.unsqueeze(0))  # (1, 1)
        return pred.squeeze(-1)                   # scalar

# ============================================================
# 6. Load best checkpoints (no training)
# ============================================================
uni_model = TextRegressor().to(device)
uni_model.load_state_dict(torch.load("best_unimodal.pt", map_location=device))
uni_model.eval()

multi_model = MultiModalRegressor(len(num_cols)).to(device)
multi_model.load_state_dict(torch.load("best_multimodal.pt", map_location=device))
multi_model.eval()

# ============================================================
# 7. Evaluation helpers — return y_true, y_pred
# ============================================================
def eval_unimodal(model, loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].squeeze(0).to(device)         # (C, 512)
            mask = batch["attention_mask"].squeeze(0).to(device)   # (C, 512)
            label = batch["label"].to(device)                      # scalar

            pred = model(ids, mask)
            y_true.append(label.item())
            y_pred.append(pred.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return y_true, y_pred, mae


def eval_multimodal(model, loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].squeeze(0).to(device)
            mask = batch["attention_mask"].squeeze(0).to(device)
            numeric = batch["numeric"].to(device)
            label = batch["label"].to(device)

            pred = model(ids, mask, numeric)
            y_true.append(label.item())
            y_pred.append(pred.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return y_true, y_pred, mae

# ============================================================
# 8. Run evaluation
# ============================================================
print("\n===== EVALUATE BEST MODELS (TEST SET) =====")
uni_y_true, uni_y_pred, uni_mae = eval_unimodal(uni_model, test_loader)
multi_y_true, multi_y_pred, multi_mae = eval_multimodal(multi_model, test_loader)

# 두 모델이 같은 세션 순서인지 체크 (문제 없으면 True)
assert np.allclose(uni_y_true, multi_y_true)
y_true = multi_y_true

print(f"Unimodal  Test MAE: {uni_mae:.4f}")
print(f"Multimodal Test MAE: {multi_mae:.4f}")
print(f"Δ (Uni - Multi): {uni_mae - multi_mae:.4f}")
print("=========================================\n")

# ============================================================
# 9. Global error analysis (Multimodal) — SAVE FIGS
# ============================================================

# 9.1 Predicted vs True (Multimodal)
plt.figure(figsize=(6, 6))
plt.scatter(y_true, multi_y_pred, alpha=0.5, s=20)

min_val = min(y_true.min(), multi_y_pred.min())
max_val = max(y_true.max(), multi_y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("True NPS")
plt.ylabel("Predicted NPS")
plt.title("Predicted vs. True NPS (Multimodal, Test Set)")
plt.grid(True)
plt.tight_layout()
pred_vs_true_path = os.path.join(FIG_DIR, "pred_vs_true_multimodal.png")
plt.savefig(pred_vs_true_path, dpi=300, bbox_inches="tight")
plt.close()

# 9.2 Residuals (Multimodal)
residuals = multi_y_pred - y_true

# (a) Residual histogram
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Residual (Predicted NPS - True NPS)")
plt.ylabel("Count")
plt.title("Residual Distribution (Multimodal, Test Set)")
plt.axvline(0, linestyle="--")
plt.tight_layout()
res_hist_path = os.path.join(FIG_DIR, "residual_hist_multimodal.png")
plt.savefig(res_hist_path, dpi=300, bbox_inches="tight")
plt.close()

# (b) Residual boxplot
plt.figure(figsize=(4, 6))
plt.boxplot(residuals, vert=True)
plt.ylabel("Residual (Predicted NPS - True NPS)")
plt.title("Residual Boxplot (Multimodal, Test Set)")
plt.axhline(0, linestyle="--")
plt.tight_layout()
res_box_path = os.path.join(FIG_DIR, "residual_box_multimodal.png")
plt.savefig(res_box_path, dpi=300, bbox_inches="tight")
plt.close()

print("Figures saved to:")
print("  -", pred_vs_true_path)
print("  -", res_hist_path)
print("  -", res_box_path)
print("\nDone.")
