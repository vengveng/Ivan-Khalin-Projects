# Empirical Methods in Finance

## Project: Dynamic Allocation and VaR of a Portfolio

### Group 09:

- Ivan Khalin
- Litian Zhang

---

## Part 1: Static Allocation

### Overview:

- Optimal portfolio weights were derived using the mean-variance criterion:
  - Incorporating stock, bond, and risk-free assets.
  - Risk aversion parameter (λ) governs the weights:
    - λ = 2: Higher allocation to risky assets.
    - λ = 10: More conservative allocations.

### Key Findings:

- Optimal weights are highly sensitive to input data, requiring careful handling of missing and extreme values.
- Static weights remain constant throughout the sample period, unlike dynamic strategies.

---

## Part 2: GARCH Model Estimation

### Non-Normality of Returns:

- Kolmogorov-Smirnov test confirmed that returns (stocks and bonds) do not follow a normal distribution.
- Ljung-Box test revealed significant autocorrelation in excess returns but not in squared excess returns.

### AR(1) Model:

- Captured some return dynamics for stocks but was less effective for bonds.
- AR(2) model for bonds showed minimal improvement, confirming AR(1) sufficiency.

### GARCH(1,1) Model:

- Modeled conditional volatility for stocks and bonds:
  - High volatility persistence (α + β ≈ 0.96).
  - Stocks exhibited greater sensitivity to shocks (higher α).

---

## Part 3: Dynamic Allocation

### Methodology:

- Dynamic strategies incorporated AR(1) expected returns and GARCH(1,1) conditional volatility.
- Non-constant risk-free rates were used.

### Performance:

- **Dynamic λ = 2**: Achieved the highest cumulative returns but with significant risk.
- **Dynamic λ = 10**: Balanced risk and return, offering the best Sharpe ratio.
- Strategies scaled proportionally with λ, emphasizing risk aversion's role in portfolio design.

### Transaction Costs:

- Dynamic strategies incur higher costs due to frequent rebalancing.
- Calculated transaction cost thresholds to match static strategy returns.

---

## Part 4: Value at Risk (VaR) Analysis

### Approaches:

1. **Unconditional VaR**:

   - Based on sample mean and variance.
   - Provided limited precision in risk modeling.

2. **Conditional VaR (GARCH)**:

   - Captured time-varying volatility but underestimated extreme risks due to assumed normality.

3. **Extreme Value Theory (GEV)**:
   - Modeled fat-tailed distributions for losses.
   - More accurate in estimating extreme risk scenarios.

### Key Metrics:

- Conditional VaR highlighted periods of heightened risk, while GEV VaR captured fat-tailed behavior.
- Risk-seeking portfolios (λ = 2) had higher VaR compared to conservative ones (λ = 10).

---

## Conclusion

- **Static Strategies**:

  - Simple and low-cost but less adaptive to market changes.
  - Suitable for highly risk-averse investors.

- **Dynamic Strategies**:

  - Outperformed static strategies in returns but involved higher risk and transaction costs.
  - Best suited for investors balancing risk and return.

- **VaR Modeling**:
  - GEV VaR provided the most reliable risk estimates during extreme market events.
  - GARCH VaR was effective for general risk assessment but underestimated tail risk.

---

## Figures

Include relevant charts for:

1. **Portfolio Weights (Static vs. Dynamic)**.
2. **Cumulative Returns for Different λ**.
3. **Evolution of VaR (GARCH vs. GEV)**.
