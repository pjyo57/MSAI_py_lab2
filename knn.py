import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def cosine_similarity(a, b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

def distances_to_all(query, X_train):
    return np.sqrt(np.sum((X_train - query) ** 2, axis=1))

def knn_predict(query, X_train, y_train, k):
    distances = distances_to_all(query, X_train)
    nearest_index = np.argsort(distances)[:k]
    labels = y_train[nearest_index]
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]

def predict_grid(grid, X_train, y_train, k):

    predictions = []

    for point in grid:
        predictions.append(knn_predict(point, X_train, y_train, k))

    return np.array(predictions)
