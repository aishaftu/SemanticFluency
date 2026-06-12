import pandas as pd

# -----------------------------
# 1. Load scheme file directly
# -----------------------------

scheme_df = pd.read_csv(
    "animals_snafu_scheme.csv",
    comment="#",
    header=None,
    names=["category", "word"]
)

scheme_df["word"] = scheme_df["word"].astype(str).str.lower().str.strip()
scheme_df["category"] = scheme_df["category"].astype(str).str.lower().str.strip()

animal_to_categories = (
    scheme_df
    .groupby("word")["category"]
    .apply(set)
    .to_dict()
)

print(f"Scheme loaded: {len(scheme_df)} entries, {scheme_df['word'].nunique()} unique animals")

# -----------------------------
# 2. Load fluency data
# -----------------------------
# Required CSV format:
# id,order,word

df = pd.read_csv("HC_Cleaned_Responses_Animals.csv")

# Restructure data: convert from wide format (Response_01, Response_02, ...) to long format
# with columns: Participant, order, word
data_records = []
for idx, row in df.iterrows():
    participant_id = row['Participant']
    for order, col in enumerate(df.columns[1:], start=1):
        word = row[col]
        # Skip NA and empty values
        if pd.notna(word) and str(word).strip() and str(word).strip().upper() != 'NA':
            data_records.append({
                'id': participant_id,
                'order': order,
                'word': str(word).strip()
            })

df = pd.DataFrame(data_records)
df["word"] = df["word"].astype(str).str.lower().str.strip()

print(f"Fluency data loaded: {df['id'].nunique()} participants, {len(df)} total responses")

# -----------------------------
# 3. Score clustering/switching
# -----------------------------

def score_participant(group):
    group = group.sort_values("order").copy()

    # remove repetitions/perseverations
    group = group.drop_duplicates(subset="word", keep="first")

    words = group["word"].tolist()
    categories = [animal_to_categories.get(w, set()) for w in words]

    switches = 0
    cluster_ids = []
    current_cluster = 1

    for i in range(len(words)):
        if i == 0:
            cluster_ids.append(current_cluster)
            continue

        shared = len(categories[i].intersection(categories[i - 1])) > 0

        if shared:
            cluster_ids.append(current_cluster)
        else:
            switches += 1
            current_cluster += 1
            cluster_ids.append(current_cluster)

    temp = group.copy()
    temp["cluster_id"] = cluster_ids

    cluster_sizes = temp.groupby("cluster_id").size()
    real_clusters = cluster_sizes[cluster_sizes >= 2]

    unknown_words = [
        w for w, cats in zip(words, categories)
        if len(cats) == 0
    ]

    return pd.Series({
        "total_correct": len(words),
        "switches": switches,
        "n_clusters": len(real_clusters),
        "mean_cluster_size": (real_clusters - 1).mean() if len(real_clusters) > 0 else 0,
        "unknown_words": ", ".join(unknown_words)
    })

scores = (
    df.groupby("id")
    .apply(score_participant)
    .reset_index()
)

scores.to_csv("animal_fluency_scores_troyer.csv", index=False)

print("\nScores:")
print(scores.to_string(index=False))
print(f"\nSaved to: animal_fluency_scores_troyer.csv")
