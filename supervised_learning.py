# -*- coding: utf-8 -*-

"""
############################################################
LIBRARIES
############################################################
"""
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    average_precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    silhouette_score,
    classification_report
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from yellowbrick.cluster import (KElbowVisualizer, SilhouetteVisualizer)

import matplotlib.pyplot as plt


"""
############################################################
DATA MANIPULATION
############################################################
"""
dataset = datasets.load_iris()

data = dataset.data
data_target = dataset['target']  # access values by key
data_target_names = dataset.target_names  # access values by attribute

print(f"Data keys:\n {dataset.keys()}")
print(f"Data:\n {data}")
print(f"Data target:\n {data_target}")
print(f"Data target names:\n {data_target_names}")

mean = statistics.mean(data[:, 0])
max_value = max(data[:, 0])
min_value = min(data[:, 0])

print(f"First column mean: {mean}")
print(f"First column max value: {max_value}")
print(f"First column min value: {min_value}")

# Splitting data
train_x1 = data[0::2, :]  # Store every 2nd row starting from the first row
train_y1 = data_target[0::2]

test_x1 = data[1::2, :]  # Store every 2nd row starting from the 2nd row
test_y1 = data_target[1::2]

print(f"train_x1:\n {train_x1} \n test_x1:\n {test_x1}")

train_X, test_X, train_y, test_y = train_test_split(data, data_target, test_size=0.3, random_state=50)
count_unique_species = np.unique(test_y, return_counts='True')

print(f"train_X velicina:\n {train_X.shape} \ntrain_y velicina:\n {train_y.shape}\n")
print(f"Number of species in test data: {count_unique_species}\n")
print(f"Number of species in test data (alt way): setosa - {np.count_nonzero(test_y == 0)}\n")
print(f"train_X:\n {train_X} \ntrain_y:\n {train_y}")

# Generate fake data
whole_numbers = np.random.randint(10, 20, size=10)  # Generate 10 whole numbers from 10 to 20
print(whole_numbers)

decimal_numbers = np.random.uniform(10, 20, size=10)  # Generate 10 decimal numbers from 10 to 20
print(decimal_numbers)

cord_points = np.random.randint(1, 100, size=(15, 2))  # Generate 15 [x, y] points to use in coordinate system.
print(cord_points)


"""
############################################################
LINEAR REGRESSION
############################################################
"""
# 1. Load and prepare data
dataset = datasets.fetch_openml('bodyfat', version=1, parser='auto')
data = dataset.data
data = data.dropna(subset=['Height', 'Weight'])

# 1.1. If multivariant linear regression
# _X = data.iloc[50:100, [3,4,5]]
# _y = data.iloc[50:100, [0]]
# X = normalize(_X)
# y = _y / np.amax(_y)

X = data['Height'].values.reshape(-1, 1) * 2.54
y = data['Weight'].values * 0.46

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 3. Train the model
modelLR = LinearRegression().fit(X_train, y_train)

# 4. Predict dependent variable
y_predict = modelLR.predict(X_test)

# 4.1. Plot results
plt.scatter(X_test, y_test, label='Real values')
plt.plot(X_test, y_predict, color='r', label='Linear regression')
plt.legend(), plt.show()

# 5. Evaluate results
print(f'Coefficient: {modelLR.coef_[0]}\n')
print(f'Intercept: {modelLR.intercept_}\n')
print(f'Coefficient of determination: {modelLR.score(X_test, y_test)}\n')
print(f'Variance: {mean_squared_error(X_test, y_test)}\n')
print(f'Value to predict: {modelLR.predict([[196]])}\n')


"""
############################################################
CLASSIFICATION K-NEAREST NEIGHBORS
############################################################
"""
# 1. Load and prepare data
dataset = datasets.load_iris()
data = dataset.data
target = dataset.target
target_names = dataset.target_names

X = data
y = target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 3. Train the model
modelKNN = KNeighborsClassifier(n_neighbors=3)
modelKNN.fit(X_train, y_train)

# 4. Predict dependent variable
y_predict = modelKNN.predict(X_test)

# 5. Evaluate results
print(f'Coefficient: {modelKNN.score(X_test, y_test)}\n')
print(f'Prediction score: {accuracy_score(y_test, y_predict)}\n')
print(f'Confusion matrix:\n{confusion_matrix(y_test, y_predict)}\n')
print(f'Classes precision:\n{precision_score(y_test, y_predict, average=None)}\n')
print(f'Model precision: {precision_score(y_test, y_predict, average="micro")}\n')
print(f'Recall score:\n{recall_score(y_test, y_predict, average=None)}\n')
print(f'F1 score:\n{f1_score(y_test, y_predict, average=None)}\n')
print(f'Classification report: {classification_report(y_test, y_predict, target_names=target_names)}\n')

"""
############################################################
KMEANS GROUPING
############################################################
"""
# 1. Load data
data = np.random.uniform(0, 50, size=(100, 2))

data[35:60] += 60
data[61:] += 130

plt.scatter(data[:, 0], data[:, 1])
plt.show()

# 2. Model
modelKM = KMeans(n_clusters=3, random_state=42, n_init=1)
modelKM.fit(data)

# 3. Predictions
groups = modelKM.predict(data)

predict_group_for_custom_value = [[297, 103]]
predicted_group = modelKM.predict(predict_group_for_custom_value)
print(f'Predicted group: {predicted_group}\n\n')

# 4. Display results
centroids = modelKM.cluster_centers_
print(f'Centroids:\n{centroids}\n\n')

plt.scatter(data[:, 0], data[:, 1], c=groups, cmap='cool')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='r', s=200)
plt.show()

# 5. Evaluation
print(f'\n\n')
visualizer = KElbowVisualizer(modelKM, k=(2, 11))
visualizer.fit(data)
visualizer.show()

print(f'\n\n')
modelKM = KMeans(n_clusters=3, random_state=42, n_init=1)
visualizer = SilhouetteVisualizer(modelKM, colors='yellowbrick')
visualizer.fit(data)
visualizer.show

silhouetter_score = metrics.silhouette_score(data, groups, metric='euclidean')
print(f'Silhouetter Score: {silhouetter_score}\n\n')


"""
############################################################
KMEANS GROUPING - IMAGE SEGMENTATION
############################################################
"""
# 1. Load and prepare data
image = imread('/content/konj1.jpg')

X = image.reshape(-1, 3)

# 3. Train the model
modelKM = KMeans(n_clusters=2, random_state=1, n_init=1)
modelKM.fit(X)

# 4. Image segmentation
labels = modelKM.labels_
centroids = modelKM.cluster_centers_

image_segmented = centroids[labels].astype(int)
image_colored = image_segmented.reshape(image.shape)

plt.imshow(image_colored)
plt.show()

# 5. Evaluate results
print(f'\n\n')
visualizer = KElbowVisualizer(modelKM, k=(2, 11))
visualizer.fit(data)
visualizer.show()

modelKM = KMeans(n_clusters=2, random_state=1, n_init=1)
visualizer = SilhouetteVisualizer(modelKM, colors='yellowbrick')
visualizer.fit(data)
visualizer.show
