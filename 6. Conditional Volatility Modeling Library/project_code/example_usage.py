import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from garch_lib import CVM
import os

# Load and preprocess data
root = os.path.dirname(__file__)
info_path = os.path.join(root, 'data', 'info.csv')
data = pd.read_csv(info_path, index_col=0, usecols=[0, 1, 2, 3], skiprows=1, parse_dates=False, names=['date', 'DAX', 'S&P', 'rate'])
data.index = pd.DatetimeIndex(data.index, freq='W-MON')
data = data[data.index.year < 2024]
data['rate'] = data['rate'] / 100 / 52
returns = data.pct_change().dropna()

# Compute weekly returns and residuals
weekly_returns = returns[['DAX', 'S&P']]
dax_residuals = AutoReg(weekly_returns['DAX'], lags=1).fit().resid
sp_residuals = AutoReg(weekly_returns['S&P'], lags=1).fit().resid

# Univariate GARCH model
print("Univariate GARCH Model:")
dax_garch = CVM('garch', 'studentt')
dax_results = dax_garch.fit(dax_residuals)
print(dax_results)

# Calculate Value-at-Risk (VaR)
dax_var = dax_garch.calc_var(dax_results)
print(f"DAX Value-at-Risk: {dax_var}")

# Multivariate DCC-GARCH model
print("\nMultivariate DCC-GARCH Model:")
combined_residuals = pd.concat([dax_residuals, sp_residuals], axis=1)
combined_residuals.columns = ['DAX', 'S&P']
dcc_model = CVM('garch', 'studentt')
dcc_results = dcc_model.fit(combined_residuals, multivar='dcc')
print(dcc_results)

# Extract correlation structure
correlation_matrices = dcc_results.correlation_structure
# print("\nCorrelation Matrices:")
# print(correlation_matrices)