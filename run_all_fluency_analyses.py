import pandas as pd
import os

# Files to process
files_to_process = [
    "HC_Cleaned_Responses_Animals.csv",
    "HC_Cleaned_Responses_Food.csv",
    "HC_Cleaned_Responses_jobs.csv",
    "SeLECTS_Cleaned_Responses_Animals.csv",
    "SeLECTS_Cleaned_Responses_Food.csv",
    "SeLECTS_Cleaned_Responses_jobs.csv",
]

# Determine scheme file based on input filename
def get_scheme_file(input_filename):
    if "Animals" in input_filename:
        return "animals_snafu_scheme.csv"
    elif "Food" in input_filename:
        return "food_snafu_scheme.csv"
    elif "jobs" in input_filename:
        return "jobs_snafu_scheme.csv"
    return "animals_snafu_scheme.csv"  # Default fallback

# Load scheme file
def load_scheme(scheme_path):
    scheme_df = pd.read_csv(
        scheme_path,
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
    return animal_to_categories, scheme_df

# Reshape wide format to long format
def reshape_data(df):
    """Convert from wide format (Response_01, Response_02, ...) to long format (id, order, word)"""
    participant_col = df.columns[0]  # Usually 'Participant' or similar
    response_cols = [col for col in df.columns if col.startswith('Response_')]
    
    # Melt from wide to long
    melted = df.melt(
        id_vars=[participant_col],
        value_vars=response_cols,
        var_name='response_order',
        value_name='word'
    )
    
    # Extract order number from response column name
    melted['order'] = melted['response_order'].str.extract('(\d+)').astype(int)
    
    # Rename participant column to 'id' and prepare dataframe
    melted = melted.rename(columns={participant_col: 'id'})
    melted['word'] = melted['word'].astype(str).str.lower().str.strip()
    
    # Remove spaces from multi-word entries (e.g., "polar bear" -> "polarbear")
    melted['word'] = melted['word'].str.replace(' ', '', regex=False)
    
    # Remove empty/NA/null responses
    melted = melted[melted['word'].notna()]
    melted = melted[melted['word'] != 'nan']
    melted = melted[melted['word'] != '']
    melted = melted[melted['word'].str.lower() != 'na']
    
    return melted[['id', 'order', 'word']].sort_values(['id', 'order'])

# Score participant function
def score_participant(group, animal_to_categories):
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

# Process each file
all_results = []

for input_file in files_to_process:
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}")
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        continue
    
    # Get appropriate scheme file
    scheme_file = get_scheme_file(input_file)
    
    if not os.path.exists(scheme_file):
        print(f"⚠️  Scheme file not found: {scheme_file}")
        print(f"   Using animals_snafu_scheme.csv as fallback")
        scheme_file = "animals_snafu_scheme.csv"
    
    # Load scheme
    print(f"Loading scheme: {scheme_file}")
    animal_to_categories, scheme_df = load_scheme(scheme_file)
    print(f"Scheme loaded: {len(scheme_df)} entries, {scheme_df['word'].nunique()} unique categories")
    
    # Load fluency data
    print(f"Loading fluency data: {input_file}")
    raw_df = pd.read_csv(input_file)
    
    # Reshape from wide to long format
    print(f"Reshaping data from wide to long format...")
    df = reshape_data(raw_df)
    
    print(f"Fluency data loaded: {df['id'].nunique()} participants, {len(df)} total responses")
    
    # Score participants
    scores = (
        df.groupby("id")
        .apply(score_participant, animal_to_categories=animal_to_categories)
        .reset_index()
    )
    
    # Add source file info
    scores["source_file"] = input_file
    
    # Save scores
    output_filename = input_file.replace(".csv", "_scores.csv")
    scores.to_csv(output_filename, index=False)
    print(f"✓ Scores saved to: {output_filename}")
    
    print("\nScores Preview:")
    print(scores.to_string(index=False))
    
    # Collect for combined report
    all_results.append({
        "source_file": input_file,
        "n_participants": df['id'].nunique(),
        "n_responses": len(df),
        "n_unique_words": df['word'].nunique(),
        "mean_total_correct": scores["total_correct"].mean(),
        "mean_switches": scores["switches"].mean(),
        "mean_n_clusters": scores["n_clusters"].mean(),
    })

# Generate summary report
print(f"\n{'='*60}")
print("SUMMARY REPORT")
print(f"{'='*60}")

summary_df = pd.DataFrame(all_results)
print(summary_df.to_string(index=False))

summary_df.to_csv("fluency_analysis_summary.csv", index=False)
print(f"\n✓ Summary saved to: fluency_analysis_summary.csv")
