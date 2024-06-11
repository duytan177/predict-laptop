import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y

    def predict(self, X):
        predicted_label = [self.__predict(x) for x in X]
        return np.array(predicted_label)

    def __predict(self, x):
        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get labels of k nearest neighbors
        k_nearest_label = [self.y_train[i] for i in k_indices]
        # Return the most common label
        return Counter(k_nearest_label).most_common(1)[0][0]

# Generate sample data
np.random.seed(1)
X = np.random.rand(15, 2) * 10
y = np.array(["tot", "xau", "xau", "tot", "tot", "xau", "tot", "tot", "tot", "xau", "tot", "xau", "xau", "xau", "tot"])

# Create an instance of the KNN class
knn = KNN(k=3)
# Train the model
knn.fit(X, y)

# Generate new data point for prediction
X_new = np.array([[5, 8]])
# Predict the label for the new data point
y_pred = knn.predict(X_new)
# Print the predicted label
print("Predicted label for the new data point:", y_pred)

# Plotting
fig, ax = plt.subplots()
colors = np.where(y == "tot", "r", "k")
ax.scatter(X[:, 0], X[:, 1], c=colors)
ax.scatter(X_new[:, 0], X_new[:, 1], c='b')

# Calculate distances to plot the nearest points
distances = [euclidean_distance(X_new[0], x_train) for x_train in X]
k_nearest_indices = np.argsort(distances)[:knn.k]
k_nearest_points = X[k_nearest_indices]

for point in k_nearest_points:
    ax.plot([X_new[0, 0], point[0]], [X_new[0, 1], point[1]], "g--")

# Calculate the radius
radius = euclidean_distance(X_new[0], k_nearest_points[-1])
print("Radius:", radius)

# Plot a circle around the new data point
circle = plt.Circle(X_new[0], radius, color="g", fill=False)
ax.add_artist(circle)

plt.show()
