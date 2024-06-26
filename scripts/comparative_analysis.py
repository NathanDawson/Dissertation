import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from paretoset import paretoset
from collections import Counter


# Load Results
results = pd.read_pickle('data/results/final_pipeline_results.pkl.bz2', compression='bz2')
results_df = pd.DataFrame(results)
results_df['Selection Technique'] = results_df['Selection Technique'].replace('Forward Feature Selection', 'FFS')
results_df['Number of Features'] = results_df['Selected Features'].apply(len)


# Aggregate data for table
table_data = results_df.groupby('Selection Technique')['Number of Features'].apply(list).reset_index()
table_data['Number of Features per Fold'] = table_data['Number of Features'].apply(lambda x: ", ".join(map(str, x)))
table_data = table_data.drop(columns=['Number of Features'])

# Create table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
plt.title('Number of Features per Fold by Selection Technique')
plt.savefig('results/feature_selection_table.png', dpi=300, bbox_inches='tight')
plt.close()


# Plot basic distributions (box-plot with swarm plot)
def plot_basic_distributions(df, feature, title, filename):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Selection Technique', y=feature, hue='Selection Technique')
    sns.swarmplot(data=df, x='Selection Technique', y=feature, color='black', alpha=0.7)
    plt.title(title)
    plt.ylabel(feature)
    plt.xlabel('Feature Selection Technique')
    plt.savefig(f'results/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


plot_basic_distributions(results_df, 'F1 Score', 'Distribution of F1 Scores per Fold by Selection Technique', 'f1_distribution')
plot_basic_distributions(results_df, 'Recall', 'Distribution of Recall Scores per Fold by Selection Technique', 'recall_distribution')


# Reshape DataFrame for grouped bar plot
grouped_df = pd.melt(results_df, id_vars=['Selection Technique', 'Fold'],
                     value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
                     var_name='Metric', value_name='Score')

# Calculate mean score for each metric and selection technique
mean_scores = grouped_df.groupby(['Selection Technique', 'Metric']).mean().reset_index()

# Create grouped bar plot
plt.figure(figsize=(12, 8))
sns.barplot(data=mean_scores, x='Selection Technique', y='Score', hue='Metric')
plt.title('Average Performance Metrics per Fold by Selection Technique')
plt.ylabel('Average Score')
plt.xlabel('Feature Selection Technique')
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('results/all_metrics_bar.png', dpi=300, bbox_inches='tight')
plt.close()


# Identify Pareto optimal solutions using F1 scores
mask = paretoset(results_df[['Number of Features', 'F1 Score']], sense=["min", "max"])
pareto_optimal_data = results_df[mask]

# Plot Pareto front with F1
plt.figure(figsize=(8, 6))
plt.scatter(results_df['Number of Features'], results_df['F1 Score'], label='All Solutions', color='blue')
plt.scatter(pareto_optimal_data['Number of Features'], pareto_optimal_data['F1 Score'], label='Pareto Optimal', color='red')

for i, row in pareto_optimal_data.iterrows():
    plt.text(row['Number of Features'], row['F1 Score'], row['Selection Technique'], fontsize=9)

plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.title('Pareto Front Analysis with F1')
plt.legend()
plt.savefig('results/f1_pareto_front_test.png', dpi=300, bbox_inches='tight')


# Identify Pareto optimal solutions using Recall scores
mask = paretoset(results_df[['Number of Features', 'Recall']], sense=["min", "max"])
pareto_optimal_data = results_df[mask]

# Plot Pareto front with Recall
plt.figure(figsize=(8, 6))
plt.scatter(results_df['Number of Features'], results_df['Recall'], label='All Solutions', color='blue')
plt.scatter(pareto_optimal_data['Number of Features'], pareto_optimal_data['Recall'], label='Pareto Optimal', color='red')

for i, row in pareto_optimal_data.iterrows():
    plt.text(row['Number of Features'], row['Recall'], row['Selection Technique'], fontsize=9)

plt.xlabel('Number of Features')
plt.ylabel('Recall Score')
plt.title('Pareto Front Analysis with Recall')
plt.legend()
plt.savefig('results/Recall_pareto_front_test.png', dpi=300, bbox_inches='tight')


# Flatten features per fold
features_per_fold = [(fold, feature) for fold, features in zip(results_df['Fold'], results_df['Selected Features']) for feature in features]

# Create DataFrame with counts of each feature per fold
feature_counts = pd.DataFrame(features_per_fold, columns=['Fold', 'Feature'])
feature_counts = feature_counts.groupby(['Fold', 'Feature']).size().reset_index(name='Counts')

# Get top 5 features for each fold
top_features_per_fold = feature_counts.groupby('Fold').apply(lambda x: x.nlargest(5, 'Counts')).reset_index(drop=True)

# Create custom order for the x-axis labels
top_features_per_fold['Fold_Feature'] = top_features_per_fold['Fold'].astype(str) + ' - ' + top_features_per_fold['Feature']
custom_order = top_features_per_fold['Fold_Feature'].unique()

# Plot grouped bar chart
plt.figure(figsize=(16, 6))
sns.barplot(data=top_features_per_fold, x='Fold_Feature', y='Counts', hue='Fold', dodge=False, palette='viridis')
plt.title('Top 5 Most Frequently Selected Features per Fold')
plt.ylabel('Count')
plt.xlabel('Feature')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('results/top_features_per_fold.png', dpi=300, bbox_inches='tight')
plt.close()


# Calculate median fold performance for Recall and F1 and flatten per fold
median_performing_folds = results_df.groupby('Selection Technique').apply(lambda x: x.loc[(x['Recall'] == x['Recall'].median()) | (x['F1 Score'] == x['F1 Score'].median())])
median_performing_folds = median_performing_folds.explode('Selected Features')
median_performing_folds['Presence'] = 1

# Count frequency of each feature
median_feature_counts = Counter(median_performing_folds['Selected Features'])
median_feature_counts_df = pd.DataFrame(median_feature_counts.items(), columns=['Feature', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Plot top 20 most frequent features in the median-performing folds
plt.figure(figsize=(10, 8))
top_features = median_feature_counts_df.head(20)
sns.barplot(data=top_features, x='Frequency', y='Feature')
plt.title('Top 20 Most Frequent Features in Median-Performing Folds')
plt.savefig('results/top_20_median_performing_features.png', dpi=300, bbox_inches='tight')
plt.close()


# Create pivot table
feature_performance_pivot = results_df.pivot_table(index='Number of Features', columns='Fold', values=['Recall', 'F1 Score'], aggfunc='mean')

# Plot heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(feature_performance_pivot, annot=True, fmt=".2f", cmap='viridis')
plt.title('Heatmap of Number of Features per Fold vs Recall and F1 Score')
plt.ylabel('Number of Features')
plt.xlabel('Fold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/number_of_features_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()


# Plot F1 Score vs. Number of Features
plt.figure(figsize=(10, 5))
sns.regplot(x='Number of Features', y='F1 Score', data=results_df, color='green', marker='+')
plt.title('Correlation between Number of Features and F1 Score')
plt.xlabel('Number of Features')
plt.ylabel('Average F1 Score')
plt.savefig('results/number_of_features_f1_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot Recall vs. Number of Features
plt.figure(figsize=(10, 5))
sns.regplot(x='Number of Features', y='Recall', data=results_df, color='green', marker='+')
plt.title('Correlation between Number of Features and Recall')
plt.xlabel('Number of Features')
plt.ylabel('Average Recall')
plt.savefig('results/number_of_features_recall_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
