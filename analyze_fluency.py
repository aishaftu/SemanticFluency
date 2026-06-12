import pandas as pd
import json
import subprocess
import os

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

    # Create clusters dictionary
    clusters_dict = {}
    for cluster_id in temp["cluster_id"].unique():
        cluster_words = temp[temp["cluster_id"] == cluster_id]["word"].tolist()
        # Only include clusters with size >= 2
        if len(cluster_words) >= 2:
            # Get the shared categories for this cluster
            cluster_categories = []
            for word in cluster_words:
                if word in animal_to_categories:
                    cluster_categories.extend(list(animal_to_categories[word]))
            clusters_dict[f"Cluster {cluster_id}"] = {
                "words": cluster_words,
                "size": len(cluster_words),
                "categories": list(set(cluster_categories))
            }

    return pd.Series({
        "total_correct": len(words),
        "switches": switches,
        "n_clusters": len(real_clusters),
        "mean_cluster_size": (real_clusters - 1).mean() if len(real_clusters) > 0 else 0,
        "unknown_words": ", ".join(unknown_words),
        "clusters": json.dumps(clusters_dict)
    })

scores = (
    df.groupby("id")
    .apply(score_participant)
    .reset_index()
)

# Save full results with clusters
scores.to_csv("animal_fluency_scores_troyer.csv", index=False)

print("\n" + "="*80)
print("ANIMAL FLUENCY SCORES WITH CLUSTERS")
print("="*80)

for idx, row in scores.iterrows():
    print(f"\n{'='*80}")
    print(f"PARTICIPANT {int(row['id'])}")
    print(f"{'='*80}")
    print(f"Total correct responses: {int(row['total_correct'])}")
    print(f"Number of switches: {int(row['switches'])}")
    print(f"Number of clusters: {int(row['n_clusters'])}")
    print(f"Mean cluster size: {row['mean_cluster_size']:.2f}")
    if row['unknown_words']:
        print(f"Unknown words: {row['unknown_words']}")
    
    # Parse and display clusters
    clusters = json.loads(row['clusters'])
    if clusters:
        print(f"\nClusters ({len(clusters)}):")
        for cluster_name, cluster_data in clusters.items():
            print(f"\n  {cluster_name}:")
            print(f"    Words: {', '.join(cluster_data['words'])}")
            print(f"    Size: {cluster_data['size']}")
            print(f"    Categories: {', '.join(cluster_data['categories'])}")
    else:
        print("\nNo clusters (all responses are isolated)")

# Calculate and display means
print(f"\n{'='*80}")
print("SUMMARY STATISTICS (MEAN ACROSS ALL PARTICIPANTS)")
print(f"{'='*80}")
print(f"Mean total correct responses: {scores['total_correct'].mean():.2f}")
print(f"Mean number of switches: {scores['switches'].mean():.2f}")
print(f"Mean number of clusters: {scores['n_clusters'].mean():.2f}")
print(f"Mean cluster size: {scores['mean_cluster_size'].mean():.2f}")

print(f"\n{'='*80}")
print(f"Saved to: animal_fluency_scores_troyer.csv")
print(f"{'='*80}\n")

# Push to repository
print("Pushing results to GitHub repository...")
try:
    # Add the file
    subprocess.run(["git", "add", "animal_fluency_scores_troyer.csv"], check=True)
    
    # Commit the file
    subprocess.run(
        ["git", "commit", "-m", "Update animal fluency scores with clusters"],
        check=True
    )
    
    # Push to repository
    subprocess.run(["git", "push"], check=True)
    
    print("✓ Successfully pushed to GitHub!")
except subprocess.CalledProcessError as e:
    print(f"✗ Error pushing to GitHub: {e}")
except FileNotFoundError:
    print("✗ Git is not installed or not in PATH")
