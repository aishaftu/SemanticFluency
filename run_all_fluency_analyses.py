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

# Score participant function with category tracking
def score_participant(group, animal_to_categories):
    group = group.sort_values("order").copy()

    # remove repetitions/perseverations
    group = group.drop_duplicates(subset="word", keep="first")

    words = group["word"].tolist()
    categories = [animal_to_categories.get(w, set()) for w in words]

    switches = 0
    cluster_ids = []
    current_cluster = 1
    word_categories = []
    cluster_categories = {}  # Track which categories are in each cluster
    last_cluster_category = None  # Track category of last cluster

    for i in range(len(words)):
        if i == 0:
            cluster_ids.append(current_cluster)
            assigned_category = list(categories[i])[0] if categories[i] else "Unknown"
            word_categories.append(assigned_category)
            cluster_categories[current_cluster] = assigned_category
            last_cluster_category = assigned_category
            continue

        assigned_category = list(categories[i])[0] if categories[i] else "Unknown"
        
        # Check if current word shares a category with the LAST cluster (not just previous word)
        shared = assigned_category == last_cluster_category and assigned_category != "Unknown"

        if shared:
            # Same category as current cluster, stay in it
            cluster_ids.append(current_cluster)
        else:
            # Different category, start new cluster
            switches += 1
            current_cluster += 1
            cluster_ids.append(current_cluster)
            cluster_categories[current_cluster] = assigned_category
            last_cluster_category = assigned_category
        
        word_categories.append(assigned_category)

    temp = group.copy()
    temp["cluster_id"] = cluster_ids
    temp["category"] = word_categories

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
        "unknown_words": ", ".join(unknown_words),
        "responses_with_categories": temp.to_dict('records')  # Store full details
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
    print(f"Scheme loaded: {len(scheme_df)} entries, {scheme_df['word'].nunique()} unique words")
    
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
    
    # Save main scores file
    output_filename = input_file.replace(".csv", "_scores.csv")
    scores_export = scores[["id", "total_correct", "switches", "n_clusters", "mean_cluster_size", "unknown_words", "source_file"]].copy()
    scores_export.to_csv(output_filename, index=False)
    print(f"✓ Scores saved to: {output_filename}")
    
    # Save detailed word-by-word categorization
    detailed_filename = input_file.replace(".csv", "_word_categories.csv")
    all_words = []
    for idx, row in scores.iterrows():
        participant_id = row['id']
        if isinstance(row['responses_with_categories'], list):
            for response in row['responses_with_categories']:
                all_words.append({
                    'participant_id': participant_id,
                    'word': response['word'],
                    'order': response['order'],
                    'cluster_id': response['cluster_id'],
                    'category': response['category']
                })
    
    if all_words:
        words_df = pd.DataFrame(all_words)
        words_df.to_csv(detailed_filename, index=False)
        print(f"✓ Word categories saved to: {detailed_filename}")
    
    print("\nScores Preview:")
    print(scores_export.to_string(index=False))
    
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
