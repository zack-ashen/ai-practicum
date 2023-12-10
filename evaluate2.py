import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

# Load the CSV data
file_path = 'evaluation.csv'  # assuming the file is in the current working directory
data = pd.read_csv(file_path)

# Convert to DataFrame
df = pd.DataFrame(data)

# 1. Statistical difference between parameterized vs non-parameterized
parametrized = df[df['Parametrized'] == True]['Accuracy']
non_parametrized = df[df['Parametrized'] == False]['Accuracy']
t_stat, p_value = ttest_ind(parametrized, non_parametrized, equal_var=False)

# 2. Statistical difference between fuzzy accuracy vs accuracy
t_stat_fuzzy, p_value_fuzzy = ttest_ind(df['Fuzzy Accuracy'], df['Accuracy'], equal_var=False)

# 3. Average accuracy
average_accuracy = df['Accuracy'].mean()

# 4. Variance of the accuracy
variance_accuracy = df['Accuracy'].var()

# 5. Bar graph of the fuzzy accuracy vs accuracy with standard error
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Model", y="Accuracy")
sns.barplot(data=df, x="Model", y="Fuzzy Accuracy", color="lightblue")
plt.title("Accuracy vs Fuzzy Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.legend(["Accuracy", "Fuzzy Accuracy"])
plt.show()

# 6. Bar graph of the average accuracy when parameterized and when not parameterized with standard error
df_grouped = df.groupby('Parametrized')['Accuracy'].agg(['mean', 'sem']).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x="Parametrized", y="mean", yerr="sem", data=df_grouped)
plt.title("Average Accuracy: Parametrized vs Non-Parametrized")
plt.ylabel("Average Accuracy")
plt.xlabel("Parametrized")
plt.show()

# 6. Bar graph of the average accuracy when parameterized and when not parameterized with standard error
df_grouped = df.groupby('Parametrized')['Accuracy'].agg(['mean', 'sem']).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x="Parametrized", y="mean", data=df_grouped, capsize=.1)
plt.errorbar(df_grouped.index, df_grouped['mean'], yerr=df_grouped['sem'], fmt='none', c='black', capsize=5)
plt.title("Average Accuracy: Parametrized vs Non-Parametrized")
plt.ylabel("Average Accuracy")
plt.xlabel("Parametrized")
plt.xticks(df_grouped.index, ['False', 'True'])
plt.show()

# Return the results
print(t_stat, p_value, t_stat_fuzzy, p_value_fuzzy, average_accuracy, variance_accuracy)