import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Load the CSV data
file_path = 'evaluation.csv'  # assuming the file is in the current working directory
data = pd.read_csv(file_path)

# Prepare data for analysis
# 1. Filter data for parameterized vs non-parameterized comparison
param_data = data[data['Parametrized'] == True]
non_param_data = data[data['Parametrized'] == False]

# Group by 'Path' since we need to compare the same images
param_grouped = param_data.groupby('Path')['Accuracy'].mean()
non_param_grouped = non_param_data.groupby('Path')['Accuracy'].mean()

# For paired t-test, we need to ensure both groups have the same images
common_paths = param_grouped.index.intersection(non_param_grouped.index)
param_grouped = param_grouped[common_paths]
non_param_grouped = non_param_grouped[common_paths]

# 2. Paired t-test for parameterized vs non-parameterized
param_vs_non_param_ttest = stats.ttest_rel(param_grouped, non_param_grouped)

# 3. Paired t-test for fuzzy accuracy vs accuracy
paired_ttest_fuzzy_accuracy = stats.ttest_rel(data['Accuracy'], data['Fuzzy Accuracy'])

# 4. Average and variance of accuracy
average_accuracy = data['Accuracy'].mean()
variance_accuracy = data['Accuracy'].var()

# 5. Bar graph for fuzzy accuracy vs accuracy
accuracy_means = data[['Accuracy', 'Fuzzy Accuracy']].mean()
accuracy_sems = data[['Accuracy', 'Fuzzy Accuracy']].sem()

# 6. Bar graph for average accuracy (parameterized vs non-parameterized)
param_accuracy_mean = param_data['Accuracy'].mean()
non_param_accuracy_mean = non_param_data['Accuracy'].mean()
param_accuracy_sem = param_data['Accuracy'].sem()
non_param_accuracy_sem = non_param_data['Accuracy'].sem()

# Plotting bar graphs
plt.figure(figsize=(14, 6))

# Fuzzy Accuracy vs Accuracy
plt.subplot(1, 2, 1)
accuracy_sems = accuracy_sems.values.flatten()
accuracy_sems = data[['Accuracy', 'Fuzzy Accuracy']].sem()
sns.barplot(x=accuracy_means.index, y=accuracy_means.values, yerr=accuracy_sems.values, capsize=7)
plt.title('Fuzzy Accuracy vs Accuracy')
plt.ylabel('Mean Accuracy')

# Parameterized vs Non-Parameterized Accuracy
plt.subplot(1, 2, 2)
means_param = [param_accuracy_mean, non_param_accuracy_mean]
errors_param = [param_accuracy_sem, non_param_accuracy_sem]
sns.barplot(x=['Parameterized', 'Non-Parameterized'], y=means_param, yerr=errors_param)
plt.title('Parameterized vs Non-Parameterized Accuracy')
plt.ylabel('Mean Accuracy')

plt.tight_layout()
plt.show()

# Returning statistical test results and calculated metrics
print(param_vs_non_param_ttest, paired_ttest_fuzzy_accuracy, average_accuracy, variance_accuracy)