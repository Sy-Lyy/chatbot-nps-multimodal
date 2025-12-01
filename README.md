# Multimodal NPS Prediction from Multilingual Chatbot Logs

This repository contains the code used for my MSc thesis on predicting
post-interaction Net Promoter Score (NPS) from multilingual customer-service
chatbot logs using a multimodal transformer model (text + behavioral features).


## 1. Overview

- **Goal**  
  Predict session-level NPS for customer-service chatbot interactions by combining
  XLM-RoBERTa–based text encodings with a compact set of behavioral features
  (activeness, engagement, interaction density, livechat flag, platform flag).

- **Models**
  - **Unimodal text model**: XLM-RoBERTa with mean-pooled CLS embeddings.
  - **Multimodal model**: XLM-RoBERTa text encoder + numeric behavior branch.

- **Main finding (short)**  
  The multimodal model consistently reduces mean absolute error (MAE) compared
  to both a dummy baseline and a strong text-only baseline on the same test set.

## 2. Repository Structure

```text
├─ README.md
│
├─ src/
│   ├─ preprocess/
│   │   ├─ 01_eda_and_cleaning.py
│   │   ├─ 02_session_features.py
│   │   └─ 03_chunking_xlmr.py
│   │
│   ├─ modeling/
│   │   ├─ dataset.py
│   │   ├─ models.py
│   │   └─ train.py
│   │
│   └─ analysis/
│       ├─ main_sub_rq1.py
│       ├─ sub_rq2.py
└─      └─ sub_rq3.py





