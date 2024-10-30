import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau
import random

# Simulated function to generate synthetic metric scores
def generate_metric_scores(num_segments, num_systems):
    # Generate BLEU scores and human scores for each system at each segment
    data = {
        "system": [],
        "segment": [],
        "human_score": [],
        "bleu_score": []
    }
    for system_id in range(num_systems):
        for segment_id in range(num_segments):
            data["system"].append(f"system_{system_id + 1}")
            data["segment"].append(segment_id + 1)
            # Simulate human score between 1 and 5, add slight randomness
            data["human_score"].append(random.randint(1, 5))
            # Simulate BLEU score between 0 and 1, loosely correlated with human score
            data["bleu_score"].append(data["human_score"][-1] / 5 + random.uniform(-0.1, 0.1))
    return pd.DataFrame(data)

# Parameters for toy example
num_segments = 10  # Number of segments (sentences) per system
num_systems = 3  # Number of translation systems

# Generate synthetic evaluation data
df = generate_metric_scores(num_segments, num_systems)

# Print the synthetic data for inspection
print("Synthetic Evaluation Data:")
print(df)

# Compute system-level Pearson correlation for each system
print("\nSystem-level Pearson correlations:")
system_level_correlations = {}
for system in df['system'].unique():
    system_data = df[df['system'] == system]
    if system_data['human_score'].nunique() > 1 and system_data['bleu_score'].nunique() > 1:
        pearson_corr, _ = pearsonr(system_data['human_score'], system_data['bleu_score'])
        system_level_correlations[system] = pearson_corr
    else:
        system_level_correlations[system] = float('nan')  # Handle cases with no variation
    print(f"{system}: {system_level_correlations[system]}")

# Compute segment-level Kendall correlation across all systems and segments
print("\nSegment-level Kendall correlation (overall):")
if df['human_score'].nunique() > 1 and df['bleu_score'].nunique() > 1:
    kendall_corr, _ = kendalltau(df['human_score'], df['bleu_score'])
else:
    kendall_corr = float('nan')
print("Segment-level Kendall correlation:", kendall_corr)
