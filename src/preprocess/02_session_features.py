# Feature Engineering

###**Duration** ###
- Compute duration per session as (latest time − earliest time).

###**Turn_counts**
- The total number of conversational turns in each session, obtained from the maximum chat_order per chat_id.

### **Activeness**

- Definition
  - Proportion of user chat messages relative to all user actions.
  - Indicates whether the user is a chat-oriented, actively typing user or a more passive, click-driven user.

- Formula
: activeness = user_chats / (user_chats + user_clicks)


###**Engagement Score**

- Definition
  - Combined indicator of how long the user stayed and how many turns they took.
  - Shows the overall level of user engagement, based on how long they stayed and how many interactions occurred.

- Components
  - turn_counts — total message count
  - duration — session length in seconds

- Formula
: 0.5 * rank(turn_counts) + 0.5 * rank(duration)


###**Interaction Density**

- Definition
  - Measures how many messages the user exchanged per second.
  - Indicates the pace of the interaction — how quickly the conversation moved.

-  Formula
: interaction_density = turn_counts / duration

###**NPS_Scpre**
- Rename and convert to integer for model input.

###**type_manual_click_yn, type_pdp_click_yn, type_buy_click_yn**
- Removed for extreme imbalance.

###**create_user**
- All values are identical (“ADMIN”), so the column was removed

###**chat_help**
- Removed due to being a direct outcome signal, since they are recorded after the conversation ends.

## Duration
"""

df["chat_time"] = df["chat_time"].astype(str).str.strip()

df["chat_time"] = pd.to_datetime(df["chat_time"], dayfirst=True, errors="coerce")


chat_duration = (
    df.groupby('chat_id')['chat_time']
      .agg(['min', 'max'])
      .reset_index()
)

chat_duration['duration'] = (chat_duration['max'] - chat_duration['min']).dt.total_seconds()

print(chat_duration.head())

df = df.merge(chat_duration[['chat_id', 'duration']], on='chat_id', how='left')

print(df[['chat_id', 'duration']].head())

df.dtypes["chat_time"]

df.head()

"""## Turn_Counts

"""

turn_counts = (
    df.groupby('chat_id')['chat_order']
    .max()
    .reset_index()
    .rename(columns={'chat_order': 'turn_counts'})
)

df = df.merge(turn_counts, on ='chat_id', how = 'left')

print(df[['chat_id', 'chat_order', 'turn_counts']].head(10))

df[df['turn_counts'].isna()]

"""## User Behavioral Features
- Activeness, Engagement_score,interaction_density

"""

# Activeness
df_user = df[df['chat_role'] == 'user']

activeness_df = (
    df_user.groupby('chat_id')['user_action']
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

activeness_df['activeness'] = (
    activeness_df['chat'] / (activeness_df['click'] + activeness_df['chat'] + 1e-6)
)

df = pd.merge(df, activeness_df[['chat_id', 'activeness']], on='chat_id', how='left')

df['activeness'] = df['activeness'].fillna(0)

# Engagement score
df['engagement_score'] = (
    df['turn_counts'].rank(pct=True) * 0.5 +
    df['duration'].rank(pct=True) * 0.5
)

# interaction_density
df['interaction_density'] = df['turn_counts'] / (df['duration'] + 1e-6)

# outlier
df.replace([np.inf, -np.inf], np.nan, inplace = True)
df.fillna(0, inplace = True)

# Normaliztion
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

df['activeness_norm'] = normalize(df['activeness']).fillna(0)
df['engagement_score_norm'] = normalize(df['engagement_score']).fillna(0)
df['interaction_density_norm'] = normalize(df['interaction_density']).fillna(0)

df.head()

"""## nps_score"""

df.rename(columns={'chat_rating': 'nps_score'}, inplace=True)
df['nps_score'] = df['nps_score'].astype(int)

"""## chat_help"""

df = df.drop(columns = 'chat_help')

"""## type_manual_click_yn, type_pdp_click_yn, type_buy_click_yn"""

cols = [
    "type_manual_click_yn",
    "type_pdp_click_yn",
    "type_buy_click_yn"
]

for c in cols:
    print(f"\n===== {c} =====")
    print("Count:")
    print(df[c].value_counts(dropna=False))
    print("\nPercent (%):")
    print((df[c].value_counts(dropna=False, normalize=True) * 100).round(2))

df = df.drop(columns = ['type_manual_click_yn',
 'type_pdp_click_yn', 'type_buy_click_yn'])

"""## Creat_user"""

df["create_user"].nunique()
df = df.drop(columns="create_user")

"""# Converting to Binary Flag
###**Live chat Y/N**
- Identify Live Chat usage by checking whether act_event_code equals "LIVE_CHAT_AGT".
- Values
  - 1 = Escalated
  - 0 = Not escalated

### **Platform Web/Whatapp**
- Indicates whether the session occurred on web or whatsapp.
- Values
  - 1 = Whatsapp
  - 0 = Web
"""

df['livechat_flag'] = df['act_event_code'].astype(str).str.upper().eq("LIVE_CHAT_AGT").astype(int)

df = df.sort_values(by=['chat_id', 'chat_order'])

df["platform_flag"] = df["platform"].map({"web": 0, "whatsapp": 1})

"""

# Text Processing
- Raw chat messages are first cleaned, and then all user messages, assistant messages are aggregated.
- The conversation is also reconstructed in a turn-by-turn format using the message order."""

#  SYSTEM,SUG : Greetings and session-start/end messages automatically generated by the chatbot; identical across all sessions and repeatedly occurring.
df = df[~df["event_type"].isin(["SYSTEM", "SUG"])]

# 1) Clean text
import re

def remove_html(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    return text.strip()

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = remove_html(text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[A-Z_]+:[A-Z0-9_]+', ' ', text)
    text = re.sub(r'[#*_`~>|]', ' ', text)
    text = re.sub(r'[^\w\s?.!/,:\'"()\/-가-힣À-ÿ]', ' ', text)
    if re.fullmatch(r'\d+', text):
        return ''
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['chat_msg'] = df['chat_msg'].apply(clean_text)


# 2) USER text
df_user = (
    df[df['chat_role'] == 'user']
    .groupby('chat_id')['chat_msg']
    .apply(lambda x: ' '.join(x))
    .reset_index(name='user_text')
)

# 3) ASSISTANT text
df_assistant = (
    df[df['chat_role'] == 'assistant']
    .groupby('chat_id')['chat_msg']
    .apply(lambda x: ' '.join(x))
    .reset_index(name='assistant_text')
)

# 4) TURN-BY-TURN combined text
def build_turns(group):
    group = group.sort_values('chat_order')
    parts = []
    for role, msg in zip(group['chat_role'], group['chat_msg']):
        if not isinstance(msg, str) or msg.strip() == "":
            continue
        if role == 'user':
            parts.append("USER: " + msg)
        else:
            parts.append("ASSISTANT: " + msg)
    return " ".join(parts)

df_turn = df.groupby('chat_id').apply(build_turns).reset_index(name='text_combined')

# 6) Merge back
df = df.merge(df_turn, on='chat_id', how='left')

print(df[['text_combined']].head())

"""# Group by chat_id
- To perform session-level analysis, only one row per chat_id is kept. The livechat_flag is consolidated at the session level (set to 1 if any message triggered it), and the duplicate removal is verified for consistency.
"""

keep_cols = ['chat_id',
    'nps_score',
    'livechat_flag',
    'platform_flag',
    'text_combined',
    'activeness_norm',
    'engagement_score_norm',
    'interaction_density_norm'
]

df = df[keep_cols]

# Find columns that vary within a chat session
nunique_by_chat = df.groupby("chat_id").nunique()

cols_with_diff = nunique_by_chat.columns[(nunique_by_chat > 1).any()]
print(cols_with_diff)

# Set livechat_flag to 1 for a session if any message contains the LIVE_CHAT_AGT event
df["livechat_flag"] = (
    df.groupby("chat_id")["livechat_flag"].transform(lambda x: 1 if (x == 1).any() else 0)
)

nunique_by_chat = df.groupby("chat_id").nunique()
cols_with_diff = nunique_by_chat.columns[(nunique_by_chat > 1).any()]
print(cols_with_diff)

# Check number of duplicate chat_id rows
total_rows = len(df)
unique_chat_ids = df["chat_id"].nunique()
duplicate_count = total_rows - unique_chat_ids

print(f"Total rows: {total_rows}")
print(f"Unique chat_id count: {unique_chat_ids}")
print(f"Duplicate rows: {duplicate_count}")

# Drop duplicates safely (save result in df2, keep df1 untouched)
df_before = len(df)
df2 = df.drop_duplicates(subset="chat_id", keep="first").copy()
df2_after = len(df2)
removed_count = df_before - df2_after

print(f"Rows after drop (df2): {df2_after}")
print(f"Removed rows: {removed_count}")

# Verify consistency between counted and removed duplicates
if removed_count == duplicate_count:
    print("The number of removed rows matches the duplicate count")
else:
    print(f"Mismatch detected: {duplicate_count} duplicates found but {removed_count} rows removed.")
    print("Original df is untouched — you can safely ignore or delete df2 if needed.")

len(df2)

df2["chat_id"].is_unique

df.isna().sum()

