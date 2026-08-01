# ML Primitives Lab

This project explores three core machine learning building blocks using NumPy:

- k-Nearest Neighbors (k-NN) classification
- Gradient descent optimization
- Principal Component Analysis (PCA) via Singular Value Decomposition (SVD)

The goal is to implement each method from scratch, visualize the results, and understand the intuition behind the algorithms rather than relying on high-level library implementations.

## Project Structure

- [knn.py](knn.py) — contains the k-NN classifier, distance computations, and helper functions
- [gradient_descent.py](gradient_descent.py) — implements 1D and 2D gradient descent examples and optimization paths
- [pca.py](pca.py) — contains the PCA implementation using SVD and projection logic
- [ml_primitives_starter.ipynb](ml_primitives_starter.ipynb) — notebook template for the lab exercises
- [requirements.txt](requirements.txt) — required Python packages
- [plots/](plots/) — folder for saving generated figures

## Requirements

This project uses Python 3 and the following libraries:

- NumPy
- Matplotlib
- Jupyter Notebook (optional, for the starter notebook)

## Setup

1. Open the project folder in your terminal.
2. Create and activate a virtual environment if needed.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Files and What They Do

### [knn.py](knn.py)

This file implements the k-NN classifier. It includes:

- Euclidean distance calculation
- Cosine similarity
- Vectorized distance computation from a query point to all training samples
- Majority-vote classification using the k nearest neighbors
- Optional plotting helpers for classification boundaries

### [gradient_descent.py](gradient_descent.py)

This file focuses on optimization. It includes:

- 1D gradient descent on a simple convex function
- Comparison of multiple learning rates
- 2D gradient descent with a path plotted over a contour map

### [pca.py](pca.py)

This file implements PCA using SVD. It includes:

- Data centering
- Singular value decomposition
- Extraction of principal components
- Projection of data onto the principal component direction

## Example Usage

```python
import numpy as np
from knn import euclidean_distance, knn_predict
from gradient_descent import gradient_descent_1d
from pca import pca_via_svd

# k-NN example
X_train = np.array([[0, 0], [1, 1], [2, 2]])
y_train = np.array([0, 1, 0])
query = np.array([1.2, 1.1])
print(knn_predict(query, X_train, y_train, k=3))

# Gradient descent example
x_final, history = gradient_descent_1d(start=10.0, lr=0.1, steps=50)
print(x_final)

# PCA example
X = np.array([[1.0, 2.1], [2.0, 4.2], [3.0, 6.3]])
projected, components = pca_via_svd(X, n_components=1)
print(projected)
print(components)
```

## Running the Lab

You can complete the lab in one of two ways:

1. Use the notebook: [ml_primitives_starter.ipynb](ml_primitives_starter.ipynb)
2. Or import the functions from the Python files and create your own scripts to generate the required plots

When generating plots, save them into the [plots/](plots/) folder so the outputs are organized and easy to review.

## Notes

- The key idea in k-NN is to compute distances efficiently using vectorized NumPy operations.
- In gradient descent, the learning rate strongly affects convergence behavior; a very large rate can cause oscillation or divergence.
- PCA requires centering the data before applying SVD; otherwise the principal direction can be misleading.

## Summary

This lab gives a practical introduction to fundamental ML building blocks and helps connect theory to real implementations with NumPy and visualization.
