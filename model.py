import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection, neighbors, tree, metrics
from imblearn.over_sampling import SMOTE

from utils.charts import plot_chart_all, plot_chart_numerical, plot_chart_by_class, plot_chart_by_IO, plot_chart_by_PA500


# Read excel file (sheet name: 'Data')
impedance_measurements = pd.read_excel('data/breast_tissue.xlsx', sheet_name='Data')

impedance_measurements
impedance_measurements.shape
impedance_measurements.info()
impedance_measurements.describe()

# Univariate distribution of numerical variables in dataset
# sns.set_style('darkgrid')
plot_chart_all(impedance_measurements)
# plt.show()

# Distribution of numerical variables in the breast_tissue dataset
# sns.set_style('whitegrid', {'axes.grid' : False})
plot_chart_numerical(impedance_measurements)
# plt.show()

# Breast tissue classification based on the numerical variables
# sns.set_style('whitegrid', {'axes.grid' : False})
plot_chart_by_class(impedance_measurements)
# plt.show()

# Breast tissue class distribution by IO
plot_chart_by_IO(impedance_measurements)

# Breast tissue class distribution by PA500
plot_chart_by_PA500(impedance_measurements)

# plt.figure(figsize=(16, 8))
sns.countplot(data=impedance_measurements, x="Class")

# Convert Class column to numeric values
impedance_measurements['Class'] = impedance_measurements['Class'].replace(
    ['adi', 'car', 'con', 'fad', 'gla', 'mas'], [0, 1, 2, 3, 4, 5]
)

impedance_measurements

# Select input and output variables
input_x = impedance_measurements.iloc[:, 1:-1].values
output_y = impedance_measurements.iloc[:, -1].values

print('input_x: ', input_x)
print('output_y: ', output_y)

# Create train and test datasets
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    input_x, output_y, test_size=0.4, random_state=0)

# Fix class imbalance
# plt.figure(figsize=(16, 8))

resampler = SMOTE(random_state=0)
x_train_smote, y_train_smote = resampler.fit_resample(x_train, y_train)

sns.countplot(x=y_train_smote)

# Scale train and test datasets
scaler = preprocessing.StandardScaler()
x_trains = scaler.fit_transform(x_train_smote)
x_tests = scaler.transform(x_test)

# Fit KNN to the training set
classifier = neighbors.KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_trains, y_train_smote)

# Predict test set result (using KNN)
y_pred = classifier.predict(x_tests)

print('y_pred: ', y_pred)
print('y_test', y_test)

knn_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
knn_result = metrics.classification_report(y_test, y_pred)

print('KNN Confusion matrix:\n', knn_confusion_matrix, '\n')
print('KNN Result:\n', knn_result)

figure, knn_chart = plt.subplots(figsize=(16, 8))

knn_chart = sns.heatmap(knn_confusion_matrix, cmap='crest', annot=True, fmt='d', linewidth=.5)
knn_chart.xaxis.tick_top()

plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.title('KNN Confusion Matrix')
# plt.show()

# Fit DecisonTreeClassifier to the training set
decision_tree_classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
decision_tree_classifier.fit(x_trains, y_train_smote)

# Predict test set result (using DecisionTree)
decision_tree_y_pred = classifier.predict(x_tests)

print('decision_tree_pred: ', decision_tree_y_pred)
print('y_test', y_test)

decision_tree_confusion_matrix = metrics.confusion_matrix(y_test, decision_tree_y_pred)
decision_tree_result = metrics.classification_report(y_test, decision_tree_y_pred)

print('DecisionTree Confusion matrix:\n', decision_tree_confusion_matrix, '\n')
print('DecisionTree Result:\n', decision_tree_result)

figure, decision_tree_chart = plt.subplots(figsize=(16, 8))
decision_tree_chart = sns.heatmap(decision_tree_confusion_matrix, cmap='crest', annot=True, fmt='d', linewidth=.5)

plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.title('DecisionTree Confusion Matrix')

plt.show()
