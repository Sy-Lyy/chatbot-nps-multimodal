"""# Text Chunking
- Splits each session’s text into multiple ≤512-token chunks by grouping sentences, ensuring model-safe input length. Adds chunk_id to track each segment.

"""
df = df2.copy()
def split_long_sessions(df, tokenizer, text_col='text_combined', max_len=512):
    from tqdm import tqdm
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    new_rows = []
    # Iterate through each row (each session)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[text_col])
        sentences = nltk.sent_tokenize(text)
        current_chunk = []
        current_len = 0
        chunk_count = 1

        for sent in sentences:
            sent_tokens = tokenizer(sent)['input_ids']
            sent_len = len(sent_tokens)

            # If a single sentence exceeds max length → split by token windows
            if sent_len > max_len:
                for j in range(0, len(sent_tokens), max_len):
                    sub_text = tokenizer.decode(sent_tokens[j:j + max_len])
                    new_row = row.copy()
                    new_row[text_col] = sub_text
                    new_row['chunk_id'] = f"{row['chat_id']}_{chunk_count}"
                    new_rows.append(new_row)
                    chunk_count += 1
                continue

            # Add sentence to current chunk if within max length
            if current_len + sent_len <= max_len:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                # Save current chunk when exceeding max length
                new_row = row.copy()
                new_row[text_col] = " ".join(current_chunk)
                new_row['chunk_id'] = f"{row['chat_id']}_{chunk_count}"
                new_rows.append(new_row)
                chunk_count += 1

                # Start a new chunk
                current_chunk = [sent]
                current_len = sent_len

        # Save the final chunk
        if current_chunk:
            new_row = row.copy()
            new_row[text_col] = " ".join(current_chunk)
            new_row['chunk_id'] = f"{row['chat_id']}_{chunk_count}"
            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    return new_df

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

df= split_long_sessions(df, tokenizer, text_col='text_combined', max_len=512)

len(df)

df["token_len"] = df["text_combined"].apply(
    lambda x: len(tokenizer.encode(str(x), truncation=False))
)

df["token_len"].describe()

df.head()

import os

save_dir = "/content/drive/MyDrive/Final7/"
os.makedirs(save_dir, exist_ok=True)

pkl_path = os.path.join(save_dir, "Preprocessed_split7.pkl")

df.to_pickle(pkl_path)

csv_path = os.path.join(save_dir, "Preprocessed_split7.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

"""

---


#Target and Feature Distributions

"""

# Binary feature list
binary_cols = ['livechat_flag', 'platform_flag']

# NPS score distribution
print("\n=== NPS Score Distribution (%) ===")
print((df['nps_score'].value_counts(normalize=True) * 100).round(2))

# Proportion of 1s for binary flags
print("\n=== Binary Flags (Share of 1s in %) ===")
print((df[binary_cols].mean() * 100).round(2))

# Quantile-based bins for behavioral features
df['activeness_bin'] = pd.qcut(df['activeness_norm'], 3, labels=['Low', 'Mid', 'High'])
df['engagement_bin'] = pd.qcut(df['engagement_score_norm'], 3, labels=['Low', 'Mid', 'High'])
df['density_bin'] = pd.qcut(df['interaction_density_norm'], 3, labels=['Low', 'Mid', 'High'])

# Distribution of bins
print("\n=== Activeness Bins ===")
print(df['activeness_bin'].value_counts(normalize=True))

print("\n=== Engagement Bins ===")
print(df['engagement_bin'].value_counts(normalize=True))

print("\n=== Interaction Density Bins ===")
print(df['density_bin'].value_counts(normalize=True))
