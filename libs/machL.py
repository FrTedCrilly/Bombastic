import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class MachL:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, df, dependent_col, explanatory_cols):
        """
        Prepare the data for linear regression.
        
        :param df: pandas DataFrame containing the data
        :param dependent_col: str, name of the dependent variable column
        :param explanatory_cols: list of str, names of the explanatory variable columns
        """
        X = df[explanatory_cols]
        y = df[dependent_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        """Train the linear regression model."""
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate the model and return performance metrics."""
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {
            'MSE': mse,
            'R2': r2,
            'Coefficients': dict(zip(self.X_train.columns, self.model.coef_)),
            'Intercept': self.model.intercept_
        }

    def predict(self, X_new):
        """
        Make predictions using the trained model.
        
        :param X_new: pandas DataFrame or numpy array of new explanatory variables
        :return: numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        return self.model.predict(X_new)

    def run_analysis(self, df, dependent_col, explanatory_cols):
        """
        Run the full linear regression analysis pipeline.
        
        :param df: pandas DataFrame containing the data
        :param dependent_col: str, name of the dependent variable column
        :param explanatory_cols: list of str, names of the explanatory variable columns
        :return: dict containing model evaluation metrics
        """
        self.prepare_data(df, dependent_col, explanatory_cols)
        self.train_model()
        return self.evaluate_model()