"""---
#*The following experiments were performed in the terminal using the GPU4EDU cluster.*
---

#**Unimodal & Multimodal NPS Modeling Pipeline**
All random seeds were fixed for reproducibility, and the dataset was split at the session level: 80% of the sessions were used for training, 20% for testing, and 10% of the training portion was held out for validation. The test set was evaluated only once at the end.

- Unimodal (Text-only)
  - Based on XLM-RoBERTa, where CLS embeddings are extracted for each text chunk.
  - Chunk-level embeddings belonging to the same session are mean-pooled, and the pooled session representation is passed through a regression layer to predict the NPS score.
- Multimodal (Text + Behavioral Features)
  - Combines XLM-R CLS embeddings with normalized behavioral features (e.g., activeness, engagement, interaction density).
  - Numerical features are projected through a feed-forward network and concatenated with the text embedding at the chunk level.
  - The fused chunk representations are mean-pooled into a single session-level vector and fed into a regression layer for NPS prediction.

Training was guided by validation performance using early stopping to prevent overfitting.

# **Hyperparameter Optimization**
- Key hyperparameters-including learning rate, batch size, number of epochs, and optimizer configuration- were tuned according to validation performance.
- The final model was selected based on the best validation MAE, and its performance was subsequently assessed on the held-out test set.
"""

import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer, AutoModel
from sklearn.dummy import DummyRegressor

# 1. Reproducibility Settings

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Encode chat_id → integer
session2idx = {sid: i for i, sid in enumerate(df["chat_id"].unique())}
df["chat_int"] = df["chat_id"].map(session2idx)

# 3. Split by session
sessions = df["chat_id"].unique()

train_sess, test_sess = train_test_split(sessions, test_size=0.2, random_state=42)
train_sess, val_sess  = train_test_split(train_sess, test_size=0.1, random_state=42)

train_df = df[df["chat_id"].isin(train_sess)]
val_df   = df[df["chat_id"].isin(val_sess)]
test_df  = df[df["chat_id"].isin(test_sess)]

print(f"Train chunks: {len(train_df)}, Val chunks: {len(val_df)}, Test chunks: {len(test_df)}")

# 4. Numeric features
num_cols = [
    "livechat_flag",
    "activeness_norm",
    "engagement_score_norm",
    "interaction_density_norm",
    "platform_flag"
]

# 5. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_texts(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

# 6. Session-level Dataset  (1 session = 1 item)
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

        # chunk texts
        texts = sess["text_combined"].tolist()
        enc = tokenize_texts(texts)      # input_ids: (C,512)

        numeric = sess[self.num_cols].iloc[0].values.astype(np.float32)
        label = sess["nps_score"].iloc[0]

        return {
            "input_ids": enc["input_ids"],            # (C, 512)
            "attention_mask": enc["attention_mask"],  # (C, 512)
            "numeric": torch.tensor(numeric, dtype=torch.float32),   # (N,)
            "label": torch.tensor(label, dtype=torch.float32)
        }

# 7. DataLoaders
train_loader = DataLoader(SessionDataset(train_df, num_cols), batch_size=1, shuffle=True)
val_loader   = DataLoader(SessionDataset(val_df, num_cols), batch_size=1)
test_loader  = DataLoader(SessionDataset(test_df, num_cols), batch_size=1)

# 8. Model Definitions

# -------- Unimodal: Text-only --------
class TextRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("xlm-roberta-base")
        self.reg_head = nn.Linear(768, 1)

    def forward(self, ids, mask):
        # ids: (C, L)
        C, L = ids.shape
        out = self.encoder(ids, mask)
        cls = out.last_hidden_state[:, 0, :]    # (C,768)
        pooled = cls.mean(dim=0)                # (768,)
        pred = self.reg_head(pooled.unsqueeze(0))   # (1,1)
        return pred.squeeze(-1)                 # scalar


# -------- Multimodal --------
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
        C, L = ids.shape

        out = self.encoder(ids, mask)
        cls = out.last_hidden_state[:, 0, :]    # (C,768)

        session_emb = cls.mean(dim=0)           # (768,)

        numeric = numeric.view(-1)              #
        num_emb = self.numeric_proj(numeric.unsqueeze(0))  # (1,32)

        fused = torch.cat([session_emb, num_emb.squeeze(0)], dim=0)
        pred = self.reg_head(fused.unsqueeze(0))
        return pred.squeeze(-1)

# 9. Training loop for UNIMODAL
def train_unimodal(model, train_loader, val_loader):
    criterion = nn.L1Loss()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best = float("inf")
    patience = 3
    wait = 0

    for epoch in range(30):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optim.zero_grad()

            ids = batch["input_ids"].squeeze(0).to(device)
            mask = batch["attention_mask"].squeeze(0).to(device)
            label = batch["label"].to(device)

            pred = model(ids, mask)
            loss = criterion(pred, label)

            loss.backward()
            optim.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].squeeze(0).to(device)
                mask = batch["attention_mask"].squeeze(0).to(device)
                label = batch["label"].to(device)

                pred = model(ids, mask)
                val_loss += criterion(pred, label).item()

        val_loss /= len(val_loader)

        print(f"[UNI] Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_unimodal.pt")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("[UNI] Early stopping")
                break

# 10. Training loop for MULTIMODAL
def train_multimodal(model, train_loader, val_loader):
    criterion = nn.L1Loss()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best = float("inf")
    patience = 3
    wait = 0

    for epoch in range(30):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optim.zero_grad()

            ids = batch["input_ids"].squeeze(0).to(device)
            mask = batch["attention_mask"].squeeze(0).to(device)
            numeric = batch["numeric"].to(device)
            label = batch["label"].to(device)

            pred = model(ids, mask, numeric)
            loss = criterion(pred, label)

            loss.backward()
            optim.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].squeeze(0).to(device)
                mask = batch["attention_mask"].squeeze(0).to(device)
                numeric = batch["numeric"].to(device)
                label = batch["label"].to(device)

                pred = model(ids, mask, numeric)
                val_loss += criterion(pred, label).item()

        val_loss /= len(val_loader)

        print(f"[MULTI] Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "best_multimodal.pt")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("[MULTI] Early stopping")
                break

# 11. Test functions
def test_unimodal(model, loader):
    model.eval()
    criterion = nn.L1Loss()
    tot = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].squeeze(0).to(device)
            mask = batch["attention_mask"].squeeze(0).to(device)
            label = batch["label"].to(device)

            pred = model(ids, mask)
            tot += criterion(pred, label).item()

    return tot / len(loader)


def test_multimodal(model, loader):
    model.eval()
    criterion = nn.L1Loss()
    tot = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].squeeze(0).to(device)
            mask = batch["attention_mask"].squeeze(0).to(device)
            numeric = batch["numeric"].to(device)
            label = batch["label"].to(device)

            pred = model(ids, mask, numeric)
            tot += criterion(pred, label).item()

    return tot / len(loader)

# 12. RUN EVERYTHING

# ----- Train Unimodal -----
print("\n===== TRAIN UNIMODAL =====\n")
uni_model = TextRegressor().to(device)
train_unimodal(uni_model, train_loader, val_loader)

# Load best
uni_model.load_state_dict(torch.load("best_unimodal.pt", map_location=device))

# Test
uni_test_mae = test_unimodal(uni_model, test_loader)
print("\nUnimodal Test MAE:", uni_test_mae)



# ----- Train Multimodal -----
print("\n===== TRAIN MULTIMODAL =====\n")
multi_model = MultiModalRegressor(len(num_cols)).to(device)
train_multimodal(multi_model, train_loader, val_loader)

# Load best
multi_model.load_state_dict(torch.load("best_multimodal.pt", map_location=device))

# Test
multi_test_mae = test_multimodal(multi_model, test_loader)
print("Multimodal Test MAE:", multi_test_mae)


# ----- Comparison -----
print("\n===== COMPARISON =====")
print(f"Unimodal MAE:   {uni_test_mae:.4f}")
print(f"Multimodal MAE: {multi_test_mae:.4f}")
print(f"Δ (Uni - Multi): {uni_test_mae - multi_test_mae:.4f}")
print("====================================\n")

# 13. Dummy baseline (session-level)

# Extract true session labels for train/test
train_sessions = train_df.drop_duplicates("chat_id")
test_sessions  = test_df.drop_duplicates("chat_id")

dummy = DummyRegressor(strategy="mean")
dummy.fit(train_sessions["nps_score"].values.reshape(-1,1),
          train_sessions["nps_score"].values)

dummy_pred = dummy.predict(test_sessions["nps_score"].values.reshape(-1,1))
dummy_mae = mean_absolute_error(test_sessions["nps_score"].values, dummy_pred)

# 14. Print all results
print("\n================ RESULTS ================")
print(f"Dummy Baseline MAE:     {dummy_mae:.4f}")
print(f"Unimodal Test MAE:      {uni_test_mae:.4f}")
print(f"Multimodal Test MAE:    {multi_test_mae:.4f}")
print("=========================================\n")
