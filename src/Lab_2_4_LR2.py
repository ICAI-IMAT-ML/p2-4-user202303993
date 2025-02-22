import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.historial_coeficientes = []
        self.historial_intercepto = []
        self.historial_formula = []
    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
            #Asegurar que X es un array 2D
        """
        X debe ser una matriz en 2 dimensiones por 
        - Filas son las observaciones 
        - Columnas cada uno de los regresores 
        """
        X = np.asarray(X) #covertimos en arrey de numpy por si acaso 
        if X.ndim == 1: #si la dimension de x es 1 signifca solo hay una var indep(regresor) que puede tener varias obsevaciones
            X = X.reshape(-1, 1) #conviertes vectro en fila en matriz con una sola columna 
        #Aplicamos la fórmula de regresión múltiple (OJO  el signo  @ multiplica matrices entre ellas)
        beta = np.linalg.inv(X.T @ X) @ X.T @ y  #Formula derivada de la funcion de perdida 1)Expando matrices 2) Derivo respecto w 3)Despeko w (X^T X)^(-1) X^T y

        #Asignamos coeficientes e intercepto
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        
    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y) #numero de ouyputs, en la fomrual de las diapositivas lo llamamos N
        
        #incializamos los coeficientes y el intercepto a un valor distinto de 0 pero muy pequeño por estandarizacion 
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01 #X-1 porque primera columna es de sesgo
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Implement gradient descent (TODO)
        for epoch in range(iterations):  #cada epóca es una t en la fórmula del gradiente 
            predictions = self.predict(X[:,1:])
            error = predictions - y
            
            # TODO: Write the gradient values and the updates for the paramenters
            gradient = np.dot(error,X) #formula del gradiente(derivada parcial=sumatorio)
            #ahora multiplicamos por alfa cada uno de los parámetros 
            self.intercept -= (learning_rate/m)*gradient[0]
            self.historial_intercepto.append(self.intercept)

            self.coefficients -= (learning_rate/m)*gradient[1:]
            self.historial_coeficientes.append(self.coefficients)

             # TODO: Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = (1/m)*(sum(error**2))
                self.historial_formula.append(mse)
                print(f"Epoch {epoch}: MSE = {mse}")


    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # TODO: Predict when X is only one variable
            predictions = self.intercept + self.coefficients * X
        else:
            # TODO: Predict when X is more than one variable
            predictions = self.intercept + X @ self.coefficients
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    # TODO
    RSS = np.sum((y_true - np.mean(y_true)) ** 2) #RSS Varianza respecto a la media  
    TSS = np.sum((y_true - y_pred) ** 2) #TSS Varianza respecto a la curva  
    r_squared = 1 - (TSS / RSS)

    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}

def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        categorical_column = X_transformed[:, index] #obetenr la columna de la matriz para el indice index (en numpy para columnas matriz[;indice], filas matriz[indice])

        # TODO: Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = np.array([unique_values == val for val in categorical_column], dtype=int)


        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = np.delete(X_transformed, index, axis=1)   #eliminamos primero
        X_transformed = np.hstack((X_transformed[:, :index], one_hot, X_transformed[:, index:]))  #añdaimos la del encoding
    return X_transformed