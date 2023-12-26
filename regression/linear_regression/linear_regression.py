import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefs_ = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        if X.shape[0] != y.shape[0]:
            raise RuntimeError("X and y must have the same number of rows.")
        
        # Check that y is a column vector
        if y.shape[1] != 1:
            raise RuntimeError("y must be a column vector.")


        # Add a column of ones to X to include an intercept
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        # Least squares solution
        self.coefs_ = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        if self.coefs_ is None:
            raise RuntimeError("You must fit the model before making predictions.")
        
        # Add a column of ones to X to include an intercept
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        # Make predictions
        y_pred = X @ self.coefs_

        return y_pred