# Sustainable Aware Asset Management

## Project: Asset Allocation with a Carbon Objective

### Group S:

- Ivan Khalin
- Liting Zhang

---

## Introduction

This project explores methods for portfolio construction and decarbonization, using the MSCI Emerging Market asset universe (2000–2021). Key objectives include evaluating investment strategies, their financial implications, and carbon impacts.

### Key Methodology:

1. **Data Collection**:

   - Data sourced from multiple files for asset returns and filters were applied to exclude zero-variance assets.

2. **Preventing Out-of-Sample Bias**:

   - Optimization and evaluation were performed iteratively to prevent forward-looking bias.

3. **Portfolio Construction**:

   - Used the `CVXPY` library for numerical optimization with a focus on constraints like minimum variance and carbon targets.

4. **Performance Metrics**:
   - Evaluated annualized returns, volatility, and Sharpe ratios while accounting for dynamic portfolio rebalancing.

---

## Part 1: Standard Asset Allocation

### Key Findings:

- **Minimum Variance Portfolio (MV)**:

  - Outperformed the Value-Weighted Portfolio (VW) in annualized returns (16.14% vs. 9.58%).
  - Lower volatility and a higher Sharpe ratio (1.06 vs. 0.40).

- **Sectoral Differences**:
  - MV favored consumer staples while VW leaned toward IT and financials, explaining performance differences during market crises.

---

## Part 2: Decarbonized Portfolios

### Strategies:

1. **MV(0.5)**: A minimum variance portfolio targeting 50% carbon emission reduction.
2. **VW(0.5)**: A value-weighted portfolio with a similar carbon reduction target.

### Performance:

- Both decarbonized portfolios performed comparably to their standard counterparts, with slight improvements in Sharpe ratios.
- Carbon footprint reduction was significant but sometimes exceeded targets due to lagging emissions data.

### Trade-offs:

- Sectoral reallocation was minimal for MV(0.5) but more pronounced for VW(0.5), due to stricter carbon constraints.

---

## Part 3: Net Zero Allocation

### VW(NZ) Portfolio:

- Implemented a net-zero emissions strategy.
- Outperformed the standard VW portfolio in returns (12.16% vs. 9.58%) with no increase in volatility.

### Key Insights:

- The net-zero portfolio achieved significant long-term carbon reduction, outperforming decarbonized strategies.

---

## Conclusion

Decarbonization strategies do not compromise financial performance. The choice of the base strategy (MV or VW) is crucial, with net-zero objectives showing strong financial and environmental benefits.

---

## Appendix

### Key Metrics for All Portfolios:

| Metric            | MV     | MV(0.5) | VW     | VW(0.5) | VW(NZ) |
| ----------------- | ------ | ------- | ------ | ------- | ------ |
| Annualized Return | 16.14% | 16.22%  | 9.58%  | 12.14%  | 12.16% |
| Volatility        | 14.15% | 13.80%  | 20.74% | 20.71%  | 20.73% |
| Sharpe Ratio      | 1.06   | 1.09    | 0.40   | 0.53    | 0.53   |
| Max Drawdown      | 1.60   | 1.68    | 1.27   | 1.36    | 1.35   |
