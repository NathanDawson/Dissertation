import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr
from paretoset import paretoset
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from adjustText import adjust_text

# Load Results
results = pd.read_pickle('data/results/final_pipeline_results.pkl.bz2', compression='bz2')
results_df = pd.DataFrame(results)

# Count the number of features selected in each row
results_df['Number of Features'] = results_df['Selected Features'].apply(len)

# Swarm plot of number of features utilised
plt.figure(figsize=(12, 8))
sns.swarmplot(data=results_df, x='Fold', y='Number of Features', hue='Selection Technique')
plt.title('Number of Features by Fold for Each Feature Selection Technique')
plt.xlabel('Fold Number')
plt.ylabel('Number of Features')
plt.xticks(list(range(1, 6)))
plt.legend(title='Feature Selection Technique', loc='upper right')
plt.savefig('results/archive/number_of_features_by_fold.png', dpi=300, bbox_inches='tight')


# Explode the 'Selected Features' column to normalise the data
results_df['presence'] = 1  # Add a binary marker
feature_matrix = results_df.explode('Selected Features').pivot_table(index='Selected Features',
                                                                     columns='Fold',
                                                                     values='presence',
                                                                     fill_value=0,
                                                                     aggfunc='sum')

# Calculate the frequency of selection across folds (normalised by the number of folds)
feature_stability = feature_matrix.div(feature_matrix.sum(axis=1), axis=0)

# Plot heatmap
plt.figure(figsize=(18, 18))
sns.heatmap(feature_stability, cmap='viridis')
plt.title('Feature Stability Across Folds')
plt.ylabel('Features')
plt.xlabel('Fold')
plt.savefig('results/archive/feature_stability_heatmap.png', dpi=600, bbox_inches='tight')


# Calculate Kruskal-Wallis test statistics and p-values for all metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
test_statistics = {}
p_values = {}

for metric in metrics:
    test_statistic, p_value = stats.kruskal(*[group[metric].values for name, group in results_df.groupby('Selection Technique')])
    test_statistics[metric] = test_statistic
    p_values[metric] = p_value

# Prepare DataFrame for plotting
stat_results = pd.DataFrame({
    'Metric': metrics * 2,
    'Value': [test_statistics[metric] for metric in metrics] + [p_values[metric] for metric in metrics],
    'Type': ['Test Statistic'] * len(metrics) + ['P-Value'] * len(metrics)
})

# Plot Kruskal-Wallis
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(data=stat_results, x='Metric', y='Value', hue='Type', palette={'Test Statistic': 'blue', 'P-Value': 'red'})
plt.title('Statistical Test Results: Kruskal-Wallis')
plt.ylabel('Value')
plt.legend(title='Data Type')

# Annotate each bar with its value
for p in bar_plot.patches:
    if p.get_height() != 0.00:
        bar_plot.annotate(format(p.get_height(), '.2f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          size=9, xytext=(0, 8),
                          textcoords='offset points')

plt.tight_layout()
plt.savefig('results/archive/kruskal_wallis_test.png', dpi=300, bbox_inches='tight')


# Plot correlation coefficient between accuracy and AUC-ROC
correlation_coefficient, p_value = pearsonr(results_df['Accuracy'], results_df['Recall'])
plt.figure(figsize=(8, 6))
sns.scatterplot(data=results_df, x='Accuracy', y='Recall', hue='Fold', palette='colorblind')
plt.title(f'Accuracy vs. Recall (Correlation: {correlation_coefficient:.2f}, p-value: {p_value:.2f})')
plt.xlabel('Accuracy')
plt.ylabel('Recall')
plt.legend(title='Fold')
plt.savefig('results/archive/corr_test_accuracy_vs_recall.png', dpi=300, bbox_inches='tight')


# Filter the results to only include the ReliefF technique
relieff_results = results_df[results_df['Selection Technique'] == 'Relief']

# Extract and count features specifically for ReliefF
relieff_feature_counts = Counter(feature for feature_list in relieff_results['Selected Features'] for feature in feature_list)

# Convert the counter to a DataFrame
relieff_feature_df = pd.DataFrame(relieff_feature_counts.items(), columns=['Feature', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Plot top 20 most common features selected by ReliefF
plt.figure(figsize=(12, 8))
sns.barplot(data=relieff_feature_df.head(20), x='Frequency', y='Feature')
plt.title('Top 20 Most Selected Features by ReliefF')
plt.xlabel('Frequency of Selection')
plt.ylabel('Feature')
plt.savefig('results/archive/relief_top_20_features_selected.png', dpi=300, bbox_inches='tight')


# Calculate the average AUC-ROC and average number of features per selection technique
avg_results = results_df.groupby('Selection Technique').agg({
    'AUC-ROC': 'mean',
    'Number of Features': 'mean'
}).reset_index()

# Identify Pareto optimal solutions based on the average values
pareto_mask = paretoset(avg_results[['Number of Features', 'AUC-ROC']], sense=['min', 'max'])
pareto_optimal_avg = avg_results[pareto_mask]

texts = []
plt.figure(figsize=(8, 6))
plt.scatter(avg_results['Number of Features'], avg_results['AUC-ROC'], label='All Techniques', color='blue')
plt.scatter(pareto_optimal_avg['Number of Features'], pareto_optimal_avg['AUC-ROC'], label='Pareto Optimal', color='red')

for i, row in avg_results.iterrows():
    texts.append(plt.text(row['Number of Features'], row['AUC-ROC'], row['Selection Technique'], fontsize=9))

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.xlabel('Average Number of Features')
plt.ylabel('Average AUC-ROC Score')
plt.title('Pareto Front Analysis Using Average Values')
plt.legend()
plt.savefig('results/archive/AUC-ROC_average_pareto_front_test.png', dpi=300, bbox_inches='tight')


# Calculate the average F1 score and average number of features per selection technique
avg_results = results_df.groupby('Selection Technique').agg({
    'F1 Score': 'mean',
    'Number of Features': 'mean'
}).reset_index()

# Identify Pareto optimal solutions based on the average values
pareto_mask = paretoset(avg_results[['Number of Features', 'F1 Score']], sense=['min', 'max'])
pareto_optimal_avg = avg_results[pareto_mask]

texts = []
plt.figure(figsize=(8, 6))
plt.scatter(avg_results['Number of Features'], avg_results['F1 Score'], label='All Techniques', color='blue')
plt.scatter(pareto_optimal_avg['Number of Features'], pareto_optimal_avg['F1 Score'], label='Pareto Optimal', color='red')

for i, row in avg_results.iterrows():
    texts.append(plt.text(row['Number of Features'], row['F1 Score'], row['Selection Technique'], fontsize=9))

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.xlabel('Average Number of Features')
plt.ylabel('Average F1 Score')
plt.title('Pareto Front Analysis Using Average Values')
plt.legend()
plt.savefig('results/archive/F1_average_pareto_front_test.png', dpi=300, bbox_inches='tight')


# Plot the relationship between number of features and F1 Score for each feature selection technique
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Number of Features', y='F1 Score', hue='Selection Technique')
plt.title('Impact of Feature Count on F1 Score by Selection Technique')
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.legend(title='Feature Selection Technique')
plt.savefig('results/archive/feature_count_vs_f1.png', dpi=300, bbox_inches='tight')

# Plot the relationship between number of features and Recall for each feature selection technique
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Number of Features', y='Recall', hue='Selection Technique')
plt.title('Impact of Feature Count on Recall by Selection Technique')
plt.xlabel('Number of Features')
plt.ylabel('Recall')
plt.legend(title='Feature Selection Technique')
plt.savefig('results/archive/feature_count_vs_recall.png', dpi=300, bbox_inches='tight')


# Extract and count features across all techniques
general_feature_counts = Counter(feature for feature_list in results_df['Selected Features'] for feature in feature_list)
general_feature_df = pd.DataFrame(general_feature_counts.items(), columns=['Feature', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Plot top 20 most common features
plt.figure(figsize=(12, 8))
sns.barplot(data=general_feature_df.head(20), x='Frequency', y='Feature')
plt.title('Top 20 Most Selected Features Across All Techniques')
plt.xlabel('Frequency of Selection')
plt.ylabel('Feature')
plt.savefig('results/archive/top_20_general_features_selected.png', dpi=300, bbox_inches='tight')


# Calculate top 25% performing folds for Recall and F1
recall_cutoff = results_df['Recall'].quantile(0.75)
f1_cutoff = results_df['F1 Score'].quantile(0.75)
high_performing_folds = results_df[(results_df['Recall'] >= recall_cutoff) | (results_df['F1 Score'] >= f1_cutoff)]

high_performing_folds = high_performing_folds.explode('Selected Features')
high_performing_folds['Presence'] = 1

high_performing_feature_counts = Counter(high_performing_folds['Selected Features'])
high_performing_feature_counts_df = pd.DataFrame(high_performing_feature_counts.items(), columns=['Feature', 'Frequency']).sort_values(by='Frequency', ascending=False)


# Plot top 20 most frequent features in the high-performing folds
plt.figure(figsize=(10, 8))
top_features = high_performing_feature_counts_df.head(20)
sns.barplot(data=top_features, x='Frequency', y='Feature')
plt.title('Top 20 Most Frequent Features in High-Performing Folds')
plt.savefig('results/archive/top_20_high_performing_features.png', dpi=300, bbox_inches='tight')
