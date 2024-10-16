import pandas as pd
import numpy as np
import cvxpy as cp
import statsmodels.api as sm  # Needed for adding constant term

class TimeSeriesLinearRegression:
    def __init__(self, df, dependent_col, explanatory_cols, regression_type='full', window=None, coeff_constraints=None):
        """
        Initialize the TimeSeriesLinearRegression class.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - dependent_col (str): The name of the dependent variable column.
        - explanatory_cols (list): A list of names of the explanatory variable columns.
        - regression_type (str): The type of regression to run ('full', 'rolling', 'expanding').
        - window (int): The window size for rolling regression (required if regression_type is 'rolling').
        - coeff_constraints (list): A list specifying constraints for each explanatory variable.
          Each element can be representations of 'positive', 'negative', or 'none' constraints.

        Attributes:
        - coefficients (pd.DataFrame or pd.Series): Coefficients from the regression(s).
        - intercepts (pd.Series): Intercepts from the regression(s).
        - t_stats (pd.DataFrame or pd.Series): t-statistics from the regression(s) (not available with constraints).
        - predictions (pd.Series): Predictions of the dependent variable.
        """

        # Copy the DataFrame to avoid modifying the original data
        self.df = df.copy()
        self.dependent_col = dependent_col
        self.explanatory_cols = explanatory_cols
        self.regression_type = regression_type
        self.window = window
        self.coeff_constraints = coeff_constraints

        # Use only the specified columns
        self.df = self.df[[dependent_col] + explanatory_cols]

        # Check if the index is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
            raise ValueError("The DataFrame index must be a datetime type for time series analysis.")

        # Process coefficient constraints
        self.coeff_constraints = self._process_constraints(coeff_constraints)

        # Check if the constraints list matches the number of explanatory variables
        if self.coeff_constraints is not None:
            if len(self.coeff_constraints) != len(self.explanatory_cols):
                raise ValueError("The length of coeff_constraints must match the number of explanatory variables.")
        else:
            # If no constraints provided, set all to None (unconstrained)
            self.coeff_constraints = [None] * len(self.explanatory_cols)

        # Initialize storage for results
        self.coefficients = None
        self.intercepts = None
        self.t_stats = None
        self.predictions = None

    def _process_constraints(self, constraints):
        """
        Process the coefficient constraints to standardize the input.

        Parameters:
        - constraints (list): List of constraints in various formats.

        Returns:
        - processed_constraints (list): List of constraints standardized to 'positive', 'negative', or None.
        """
        if constraints is None:
            return [None] * len(self.explanatory_cols)

        processed_constraints = []
        for c in constraints:
            if c in [1, '+', 'pos', 'Pos', 'positive', 'Positive']:
                processed_constraints.append('positive')
            elif c in [-1, '-', 'neg', 'Neg', 'negative', 'Negative']:
                processed_constraints.append('negative')
            elif c in [0, None, 'none', 'None', 'unconstrained', 'Unconstrained']:
                processed_constraints.append(None)
            else:
                raise ValueError(f"Invalid constraint value: {c}. Allowed values are 1, '+', 'pos', 'positive', -1, '-', 'neg', 'negative', 0, None, 'none'.")
        return processed_constraints

    def run_regression(self):
        """
        Run the specified type of regression.
        """
        if self.regression_type == 'full':
            self._run_full_sample_regression()
        elif self.regression_type == 'rolling':
            if self.window is None:
                raise ValueError("Window size must be specified for rolling regression.")
            self._run_rolling_regression()
        elif self.regression_type == 'expanding':
            self._run_expanding_regression()
        else:
            raise ValueError("Invalid regression type. Choose 'full', 'rolling', or 'expanding'.")

    def _constrained_regression(self, y, X):
        """
        Perform constrained regression for a single regression using cvxpy.
        """
        n_features = X.shape[1]

        # Define cvxpy variables
        beta = cp.Variable(n_features)

        # Objective function: Minimize the sum of squared residuals
        objective = cp.Minimize(cp.sum_squares(y - X @ beta))

        # Constraints
        constraints = []

        for i, constraint in enumerate(self.coeff_constraints):
            idx = i + 1  # Adjust for intercept term at beta[0]
            if constraint == 'positive':
                constraints.append(beta[idx] >= 0)
            elif constraint == 'negative':
                constraints.append(beta[idx] <= 0)
            # If constraint is None (unconstrained), do nothing

        # Set up and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError("Optimization failed: " + problem.status)

        beta_est = beta.value

        # t-statistics are not available in constrained regression
        se_beta = None

        return beta_est, se_beta

    def _run_full_sample_regression(self):
        """
        Run a full in-sample regression using all available data.
        """
        y = self.df[self.dependent_col].values
        X = self.df[self.explanatory_cols]
        X = sm.add_constant(X)  # Add intercept term
        X_values = X.values

        beta_est, se_beta = self._constrained_regression(y, X_values)

        self.coefficients = pd.Series(beta_est[1:], index=self.explanatory_cols)
        self.intercepts = beta_est[0]
        self.t_stats = None  # t-statistics are not available in constrained regression
        self.predictions = pd.Series(X_values @ beta_est, index=self.df.index)

    def _run_rolling_regression(self):
        """
        Run a rolling regression with a specified window size.
        """
        y = self.df[self.dependent_col].values
        X = self.df[self.explanatory_cols]
        X = sm.add_constant(X)
        X_values = X.values

        coef_list = []
        intercept_list = []
        pred_list = []
        index_list = []

        for i in range(self.window - 1, len(self.df)):
            y_window = y[i - self.window + 1: i + 1]
            X_window = X_values[i - self.window + 1: i + 1, :]
            beta_est, se_beta = self._constrained_regression(y_window, X_window)
            coef_list.append(beta_est[1:])
            intercept_list.append(beta_est[0])
            prediction = X_values[i, :] @ beta_est
            pred_list.append(prediction)
            index_list.append(self.df.index[i])

        # Convert lists to DataFrames/Series
        self.coefficients = pd.DataFrame(coef_list, index=index_list, columns=self.explanatory_cols)
        self.intercepts = pd.Series(intercept_list, index=index_list)
        self.t_stats = None  # t-statistics are not available in constrained regression
        self.predictions = pd.Series(pred_list, index=index_list)

    def _run_expanding_regression(self):
        """
        Run an expanding window regression, recalculating the model at each step forward.
        """
        y = self.df[self.dependent_col].values
        X = self.df[self.explanatory_cols]
        X = sm.add_constant(X)
        X_values = X.values

        coef_list = []
        intercept_list = []
        pred_list = []
        index_list = []

        for i in range(1, len(self.df)):
            y_window = y[:i + 1]
            X_window = X_values[:i + 1, :]
            beta_est, se_beta = self._constrained_regression(y_window, X_window)
            coef_list.append(beta_est[1:])
            intercept_list.append(beta_est[0])
            prediction = X_values[i, :] @ beta_est
            pred_list.append(prediction)
            index_list.append(self.df.index[i])

        # Convert lists to DataFrames/Series
        self.coefficients = pd.DataFrame(coef_list, index=index_list, columns=self.explanatory_cols)
        self.intercepts = pd.Series(intercept_list, index=index_list)
        self.t_stats = None  # t-statistics are not available in constrained regression
        self.predictions = pd.Series(pred_list, index=index_list)
