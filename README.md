# Linear Regression Exercise

This exercise focuses on implementing various components of linear regression from scratch. The following functions need to be completed:

## Functions to Implement

### 1. Evaluation Function
`evaluate_regression(y_true, y_pred)`: Calculates regression metrics:
- R² (coefficient of determination)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

### 2. Data Preprocessing
`one_hot_encode(data)`: Converts categorical variables into one-hot encoded format

### 3. LinearRegressor Class
- `fit(self, X, y, method='gradient_descent')`: Fits the linear regression model using Gradient descent:

You will also implement several visualization functions in the jupyter notebook.

## Testing
The implementation will be verified against the following test cases:
- Basic model fitting using gradient descent
- Regression metrics calculation
- One-hot encoding transformation
