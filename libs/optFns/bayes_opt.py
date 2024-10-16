import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Example DataFrame of strategy returns
# Replace this with your actual DataFrame of time series returns
outDir = r"C:\Users\edgil\Documents\git_working\Research\pcode\tests\dummy_data\\"
pnls = pd.read_csv(outDir + "pnls.csv", index_col=0, parse_dates=True)
strategy_returns =  pnls.diff(1).fillna(0)

# Risk-free rate, assuming 0 for simplicity
R_f = 0.0
# Number of strategies
num_strategies = strategy_returns.shape[1]

# Define the search space (weights) with bounds between 0 and 1 and assign names
space = [Real(0.0, 1.0, name=f'weight_{i}') for i in range(num_strategies)]


# The objective function to be minimized
@use_named_args(dimensions=space)
def objective_function(**kwargs):
    weights = np.array(list(kwargs.values()))

    # Normalize the weights so they sum to 1
    weights /= np.sum(weights)

    portfolio_return = np.dot(weights, strategy_returns.mean())
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(strategy_returns.cov(), weights)))
    sharpe_ratio = (portfolio_return - R_f) / portfolio_volatility

    return -sharpe_ratio  # Minimize the negative Sharpe Ratio


# Perform Bayesian optimization
result = gp_minimize(func=objective_function,
                     dimensions=space,
                     n_calls=50,
                     acq_func="EI",  # Expected Improvement
                     )

# Extract the optimal weights
optimal_weights = np.array([result.x[i] for i in range(num_strategies)])

print("Optimal Weights:", optimal_weights)
# Get the names of the strategies from the columns of the DataFrame
strategy_names = strategy_returns.columns

# Map the optimal weights to the corresponding strategy names
optimal_weights_dict = dict(zip(strategy_names, optimal_weights))

print("Optimal Weights by Strategy:")
for strategy, weight in optimal_weights_dict.items():
    print(f"{strategy}: {weight:.5f}")