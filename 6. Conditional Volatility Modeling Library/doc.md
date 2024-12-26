# CVM Library Features

The `CVM` library provides tools for univariate and multivariate GARCH modeling, allowing for extensive analysis of financial time series data.

---

## 1. **Model Initialization**
   - **Specify Model Type**:
     - GARCH: `'garch'`
     - GJR-GARCH: `'gjr'`
   - **Specify Distribution**:
     - Normal: `'normal'`
     - Student-t: `'studentt'`
     - Skew-t: `'skewt'`
   - **Example**:
     ```python
     model = CVM('garch', 'studentt')
     ```

---

## 2. **Model Fitting**
   - **Univariate Fit**:
     - Fit GARCH or GJR models to residuals.
     - Example:
       ```python
       results = model.fit(dax_residuals)
       ```
   - **Multivariate Fit**:
     - Fit models to multiple time series with DCC (Dynamic Conditional Correlations).
     - Example:
       ```python
       results = model.fit(combined_df, multivar='dcc')
       ```

---

## 3. **Parameter Estimation**
   - Retrieve estimated parameters of the fitted model.
   - Example:
     ```python
     print(results)
     ```

---

## 4. **Value-at-Risk (VaR) Calculation**
   - Calculate VaR based on the fitted GARCH model.
   - Example:
     ```python
     var = model.calc_var(results)
     print("Value-at-Risk:", var)
     ```

---

## 5. **Correlation Structure (Multivariate Models)**
   - **Extract Correlation Matrices**:
     - Available after fitting a multivariate model.
     - Example:
       ```python
       correlation_matrices = results.correlation_structure
       print(correlation_matrices)
       ```

   - **Analyze Time-Series Correlations**:
     - Extract off-diagonal correlations for time-series analysis.
     - Example:
       ```python
       pairwise_correlations = [
           results.correlation_structure[t, i, j]
           for t in range(results.correlation_structure.shape[0])
           for i in range(results.correlation_structure.shape[1])
           for j in range(i + 1, results.correlation_structure.shape[1])
       ]
       ```

---

## 6. **Supported Features**
   - **Univariate Modeling**:
     - GARCH and GJR models with Normal, Student-t, or Skew-t distributions.
   - **Multivariate Modeling**:
     - Dynamic Conditional Correlation (DCC) framework with GARCH or GJR.
   - **Custom Configurations**:
     - Easy switching between univariate and multivariate modes.
   - **Output**:
     - Parameters, correlation structures, and calculated risk metrics like VaR.

---

## Example Workflow
```python
from garch_lib import CVM

# Initialize a GARCH model
model = CVM('garch', 'normal')

# Fit the model to residuals
results = model.fit(dax_residuals)

# Calculate Value-at-Risk
var = model.calc_var(results)
print("Value-at-Risk:", var)

# Multivariate example
combined_df = pd.concat([dax_residuals, sp_residuals], axis=1)
combined_df.columns = ['DAX', 'S&P']
dcc_model = CVM('garch', 'studentt')
dcc_results = dcc_model.fit(combined_df, multivar='dcc')

# Extract correlation structure
correlation_matrices = dcc_results.correlation_structure
print("Correlation Matrices:", correlation_matrices)