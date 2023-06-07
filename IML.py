import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Print the correct and wrong predictions
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print(f"Correct prediction: {X_test[i]} -> {iris.target_names[y_pred[i]]}")
    else:
        print(f"Wrong prediction: {X_test[i]} -> Predicted: {iris.target_names[y_pred[i]]}, Actual: {iris.target_names[y_test[i]]}")