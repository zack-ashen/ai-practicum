import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import sem

df = pd.read_csv("./evaluation2.csv")

# 1. Average accuracy
mean_accuracy = df['Accuracy'].mean()
std_accuracy = df['Accuracy'].std()
print(f"Mean Accuracy: {mean_accuracy}, Standard Deviation: {std_accuracy}")

# 2. Average fuzzy accuracy
mean_fuzzy_accuracy = df['Fuzzy Accuracy'].mean()
std_fuzzy_accuracy = df['Fuzzy Accuracy'].std()
print(f"Mean Fuzzy Accuracy: {mean_fuzzy_accuracy}, Standard Deviation: {std_fuzzy_accuracy}")

# 3. Average Jaccard accuracy
mean_jaccard_accuracy = df['Jaccard Accuracy'].mean()
std_jaccard_accuracy = df['Jaccard Accuracy'].std()
print(f"Mean Jaccard Accuracy: {mean_jaccard_accuracy}, Standard Deviation: {std_jaccard_accuracy}")

# 4. Average accuracy for '../photosv2/drinks2.png' and its p-value
drinks2_accuracy = df[df['Path'] == '../photosv2/drinks2.png']['Accuracy']
not_drinks2_accuracy = df[df['Path'] != '../photosv2/drinks2.png']['Accuracy']
mean_drinks2_accuracy = drinks2_accuracy.mean()
t_stat_drinks2, p_val_drinks2 = stats.ttest_1samp(drinks2_accuracy, not_drinks2_accuracy.mean())
print(f"Drinks 2 Image Accuracy: {drinks2_accuracy.mean()}, p-value: {p_val_drinks2} \n")

# 5. Average accuracy based on Parametrized value and its p-value
accuracy_param_false = df[df['Parametrized'] == False]['Accuracy']
accuracy_param_true = df[df['Parametrized'] == True]['Accuracy']
mean_acc_param_false = accuracy_param_false.mean()
mean_acc_param_true = accuracy_param_true.mean()
t_stat_param, p_val_param = stats.ttest_ind(accuracy_param_false, accuracy_param_true)
print(f"Average Accuracy for Parametrized=False: {mean_acc_param_false}")
print(f"Average Accuracy for Parametrized=True: {mean_acc_param_true}")
print(f"p-value for Parametrized=False vs Parametrized=True: {p_val_param} \n")

# Obstruction Test
obstruction_accuracy = df[df['Path'].isin(['../photosv2/711_Sodas_Stickers.png', '../photosv2/711_Juice.png', '../photosv2/711_Sodas.png'])]['Accuracy']
not_obstruction_accuracy = df[~df['Path'].isin(['../photosv2/711_Sodas_Stickers.png', '../photosv2/711_Juice.png', '../photosv2/711_Sodas.png'])]['Accuracy']
mean_obstruction_accuracy = obstruction_accuracy.mean()
t_stat_obstruction, p_val_obstruction = stats.ttest_ind(obstruction_accuracy, not_obstruction_accuracy)
print(f"Obstruction Accuracy: {obstruction_accuracy.mean()}, p-value: {p_val_obstruction} \n Not Obstruction Accuracy: {not_obstruction_accuracy.mean()}")
mean_obst_accuracies = [not_obstruction_accuracy.mean(), obstruction_accuracy.mean()]
labels = ['Obstructed=False', 'Obstructed=True']

# Adding error bars to the graph for Parametrized values
std_obs_param_false = not_obstruction_accuracy.std()
std_obs_param_true = obstruction_accuracy.std()

plt.bar(labels, mean_obst_accuracies, yerr=[sem(not_obstruction_accuracy), sem(obstruction_accuracy)], capsize=5)
plt.xlabel('Obstruction')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Comparison by Obstruction')
plt.show()


# 6. Bar graph of average accuracy for Parametrized values
mean_accuracies = [mean_acc_param_false, mean_acc_param_true]
labels = ['Parametrized=False', 'Parametrized=True']

# Adding error bars to the graph for Parametrized values
std_acc_param_false = accuracy_param_false.std()
std_acc_param_true = accuracy_param_true.std()

plt.bar(labels, mean_accuracies, yerr=[sem(accuracy_param_false), sem(accuracy_param_true)], capsize=5)
plt.xlabel('Parametrized')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Comparison by Parametrization')
plt.show()

# Creating a bar graph comparing average accuracy, average fuzzy accuracy, and average jaccard accuracy
mean_values = [mean_accuracy, mean_fuzzy_accuracy, mean_jaccard_accuracy]
sem_values = [sem(df['Accuracy']), sem(df['Fuzzy Accuracy']), sem(df['Jaccard Accuracy'])]
comparison_labels = ['Average Accuracy', 'Average Fuzzy Accuracy', 'Average Jaccard Accuracy']

plt.bar(comparison_labels, mean_values, yerr=sem_values, capsize=5)
plt.ylabel('Average Value')
plt.title('Comparison of Different Accuracy Metrics')
plt.show()

f_statistic, accuracy_p_value = stats.f_oneway(df['Accuracy'], df['Fuzzy Accuracy'], df['Jaccard Accuracy'])
print(f"p-value for Accuracy vs Fuzzy Accuracy vs Jaccard Accuracy: {accuracy_p_value} \n")