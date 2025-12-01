# Data Loading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = "data/raw/final.csv"

df = pd.read_csv(csv_path, encoding="utf-8-sig")

print(df.head())

df.columns = df.columns.str.lower()
print(df.columns.tolist())

print("Number of unique chat_id:", df["chat_id"].nunique())
"""

---


# EDA

### **Basic metadata**

- chat_id — unique ID of each chat session

- chat_order — message order within a session

- chat_time — timestamp of each message

### **User/bot interaction**

- chat_role — user or assistant (bot)

- chat_msg — text content of the message

- user_action — click/chat action performed by the user

- platform — source platform (web, WhatsApp, etc.)

### **Event & behavior logs**

- act_event_code — internal action/event code

- event_type — type of interaction event

- type_manual_click_yn, type_pdp_click_yn, type_buy_click_yn — click-type indicators

### **Client & system**

- client_ip — hashed IP information

- create_user — system-generated creator ID

- Conversation outcome

- chat_rating — post-chat customer rating

- chat_help — whether the chat was helpful

- chat_comment — customer’s written feedback"""

# ===== 1. Basic structure =====
print("=== Basic Info ===")
print(df.info())

print("\n=== Shape ===")
print(df.shape)

print("\n=== First rows ===")
print(df.head())

# ===== 2. Missing values =====
print("\n=== Missing value ratio ===")
print(df.isna().mean().sort_values(ascending=False))

# ===== 3. Column types =====
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\n=== Categorical columns ===")
print(cat_cols)

print("\n=== Numeric columns ===")
print(num_cols)

# ===== 4. Basic stats =====
print("\n=== Numerical summary ===")
print(df[num_cols].describe().T)

# ===== 5. Categorical distribution (top 10) =====
print("\n=== Categorical value counts (top 10) ===")
for col in cat_cols:
    print(f"\n-- {col} --")
    print(df[col].value_counts().head(10))

# ===== 6. Session-level stats (chat_id) =====
if "chat_id" in df.columns:
    session_len = df.groupby("chat_id").size()
    print("\n=== Session length (messages per chat_id) ===")
    print(session_len.describe())

    plt.figure(figsize=(6,4))
    plt.hist(session_len, bins=40)
    plt.title("Messages per session")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.show()

# ===== 7. Chat role distribution =====
if "chat_role" in df.columns:
    plt.figure(figsize=(5,4))
    sns.countplot(data=df, x="chat_role")
    plt.title("Chat Role Distribution")
    plt.show()

# ===== 8. Platform distribution =====
if "platform" in df.columns:
    plt.figure(figsize=(5,4))
    sns.countplot(data=df, x="platform")
    plt.title("Platform Distribution")
    plt.show()

# ===== 9. Text length distribution (chat_msg) =====
if "chat_msg" in df.columns:
    df["msg_length"] = df["chat_msg"].astype(str).str.len()

    print("\n=== chat_msg length summary ===")
    print(df["msg_length"].describe())

    plt.figure(figsize=(6,4))
    sns.histplot(df["msg_length"], bins=50)
    plt.title("chat_msg Length Distribution")
    plt.xlabel("Length")
    plt.show()

# ===== 10. Time-of-day distribution =====
if "chat_time" in df.columns:
    df["chat_time"] = pd.to_datetime(df["chat_time"], errors="coerce")
    df["hour"] = df["chat_time"].dt.hour

    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="hour")
    plt.title("Messages by Hour")
    plt.xlabel("Hour")
    plt.show()

# ===== 11. NPS / chat_rating distribution =====
if "chat_rating" in df.columns:
    print("\n=== chat_rating (NPS) summary ===")
    print(df["chat_rating"].describe())  # count, mean, std, min, 25%, 50%, 75%, max

    plt.figure(figsize=(6, 4))
    sns.histplot(df["chat_rating"].dropna(), bins=11, discrete=True)
    plt.title("NPS (chat_rating) Distribution")
    plt.xlabel("NPS score (0–10)")
    plt.ylabel("Count")
    plt.show()

"""

---


# Missing Values Handling
### **act_event_code, event_type**
- These columns are not used as model inputs; they are only used for:
  - identifying Live Chat sessions (act_event_code == "LIVE_CHAT_AGT"), and

  - filtering out automatic system-generated messages (event_type == "SYSTEM" or "SUG).

1. **Validation before handling missing values**
- Before imputing missing values, the following checks were performed:
  - Missing act_event_code rows were checked to see whether their chat_msg matched the chat_msg of any LIVE_CHAT_AGT messages.

  - Missing event_type rows were checked to see whether their chat_msg matched the chat_msg of any SYSTEM messages.

2. **Result**

- No missing entries matched these event categories,
so all remaining missing values were safely filled with "UNKNOWN".

###**client_ip**
- Removed due to no values

###**chat_msg**
- Fill missing chat messages with empty strings.

###**chat_comment**
- Removed from the modeling set because, although it contains missing values, it is a direct post-conversation outcome signal.


"""

nan_counts = df.isna().sum()
nan_percent = (df.isna().mean() * 100).round(2)

nan_summary = pd.DataFrame({
    "NaN_count": nan_counts,
    "NaN_percent": nan_percent
}).sort_values("NaN_percent", ascending=False)

print(nan_summary)

"""## act_event_code, event_type

### act_event_code
"""

# Identifying Live Chat Trigger Messages
livechat_rows = df[df["act_event_code"].astype(str).str.upper() == "LIVE_CHAT_AGT"]
livechat_msgs = livechat_rows["chat_msg"].unique()

print(livechat_msgs)
print("Unique LIVE_CHAT_AGT msg count:", len(livechat_msgs))

# Checking whether missing act_event_code rows contain live chat triggers
missing_rows_act = df[df["act_event_code"].isna()]

missing_livechat_matches = missing_rows_act[
    missing_rows_act["chat_msg"].isin(livechat_msgs)
]

print("LIVE_CHAT_AGT found in NA rows:", len(missing_livechat_matches))

# Imputing Missing act_event_code
df["act_event_code"] = df["act_event_code"].fillna("UNKNOWN")

"""### event_type"""

# Identifying SYSTEM Event Messages
system_rows = df[df["event_type"] == "SYSTEM"]
system_msgs = system_rows["chat_msg"].unique()

print(system_msgs)
print("Unique SYSTEM msg count:", len(system_msgs))

# Verifying Whether Missing Event-Code Rows Contain SYSTEM-Type Messages
missing_rows_event = df[df["event_type"].isna()]   # FIXED HERE

missing_system_matches = missing_rows_event[
    missing_rows_event["chat_msg"].isin(system_msgs)
]

print("SYSTEM found in NA rows:", len(missing_system_matches))

# Imputing Missing event_type
df["event_type"] = df["event_type"].fillna("UNKNOWN")

"""

## client_ip"""

df["client_ip"].nunique()

df = df.drop(columns = 'chat_comment')

"""## chat_msg

"""

df['chat_msg'] = df['chat_msg'].fillna('')

"""## chat_comment"""

df = df.drop(columns = 'chat_comment')

"""## Re-check missing values"""

nan_counts = df.isna().sum()
nan_percent = (df.isna().mean() * 100).round(2)

nan_summary = pd.DataFrame({
    "NaN_count": nan_counts,
    "NaN_percent": nan_percent
}).sort_values("NaN_percent", ascending=False)

print(nan_summary)

"""---
