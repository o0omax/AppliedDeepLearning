import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = './experiments'

# detect experiment folders
folders = {
    folder: os.path.join(base_dir, folder)
    for folder in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, folder))
}

def load_results_csv(folder):
    results_path = os.path.join(folder, 'results.csv')
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None


results_data = {}
for name, folder in folders.items():
    results = load_results_csv(folder)
    if results is not None:
        results_data[name] = results


summary_data = []
for name, data in results_data.items():
    last_epoch = data.iloc[-1]
    # Match column names with spaces
    precision_col = next((col for col in data.columns if 'precision' in col.strip()), None)
    recall_col = next((col for col in data.columns if 'recall' in col.strip()), None)
    map50_col = next((col for col in data.columns if 'mAP_0.5' in col.strip()), None)
    map5095_col = next((col for col in data.columns if 'mAP_0.5:0.95' in col.strip()), None)

    summary_data.append({
        'Experiment': name,
        'Precision': last_epoch[precision_col] if precision_col else np.nan,
        'Recall': last_epoch[recall_col] if recall_col else np.nan,
        'mAP_50': last_epoch[map50_col] if map50_col else np.nan,
        'mAP_50_95': last_epoch[map5095_col] if map5095_col else np.nan,
    })

summary_df = pd.DataFrame(summary_data)

# Benchmark values
benchmark_metrics = {
    'Precision': 0.698,
    'Recall': 0.594,
    'mAP_50': 0.617
}

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Bar plots
axes[0].bar(summary_df['Experiment'], summary_df['Precision'], label='Experiments')
axes[0].axhline(benchmark_metrics['Precision'], color='red', linestyle='--', label='Benchmark')
axes[0].set_title('Precision Comparison')
axes[0].set_ylabel('Precision')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()

axes[1].bar(summary_df['Experiment'], summary_df['Recall'], label='Experiments')
axes[1].axhline(benchmark_metrics['Recall'], color='red', linestyle='--', label='Benchmark')
axes[1].set_title('Recall Comparison')
axes[1].set_ylabel('Recall')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()

axes[2].bar(summary_df['Experiment'], summary_df['mAP_50'], label='Experiments')
axes[2].axhline(benchmark_metrics['mAP_50'], color='red', linestyle='--', label='Benchmark')
axes[2].set_title('mAP_50 Comparison')
axes[2].set_ylabel('mAP_50')
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend()

plt.tight_layout()
plt.show()

# Determine the best experiment based on mAP_50
if 'mAP_50' in summary_df.columns and not summary_df['mAP_50'].isna().all():
    best_experiment = summary_df.loc[summary_df['mAP_50'].idxmax()]
    print("Best Approach Based on mAP_50:")
    print(best_experiment)
else:
    print("No valid mAP_50 data found to determine the best experiment.")

# Display the summary table
print("\nSummary Table:")
print(summary_df)
