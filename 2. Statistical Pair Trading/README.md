# Empirical Methods in Finance

## Project: Cointegration and Pair Trading

### Group 09:

- Ivan Khalin
- Litian Zhang

---

## Part 1: Descriptive Statistics

### Key Observations:

1. **Log vs. Simple Returns**:

   - Log returns systematically attenuate high returns and overstate low returns.
   - Differences are negligible for daily returns but increase for weekly returns.

2. **Moments of Distributions**:
   - Log returns exhibit lower mean, minimum, and maximum values.
   - Variance is nearly identical between log and simple returns.
   - Skewness is significantly affected, with log returns often showing negative skewness.

---

## Part 2: Stationarity

### Unit Root Testing:

1. **Definitions**:

   - A time series with a unit root is non-stationary.
   - Null hypothesis: Unit root exists (𝜙 = 1).
   - Alternative hypothesis: Series is stationary (𝜙 < 1).

2. **Dickey-Fuller Test**:

   - Used to evaluate stationarity.
   - Simulated critical values closely matched theoretical values.

3. **Findings**:

   - All five log-price series failed to reject the null hypothesis, indicating non-stationarity.

4. **Cointegration Potential**:
   - Linear combinations of non-stationary series may result in stationarity, enabling pair trading strategies.

---

## Part 3: Cointegration

### Key Findings:

1. **Spurious Regressions**:

   - Simulated cointegration tests for random walks emphasized the importance of critical values to avoid false positives.

2. **Asset Pair Analysis**:
   - Positive linear relationships were observed between asset pairs, likely due to similar industry influences.
   - IHG-MAR pair exhibited the strongest cointegration, suggesting high synchronization in price movements.

---

## Part 4: Pair Trading

### Strategy Design:

1. **Mean-Reverting Spreads**:

   - Cointegrated asset pairs have spreads that revert to the mean, enabling statistical arbitrage through long-short positions.

2. **Spread Normalization**:

   - Normalized spreads are unitless and directly represent standard deviations from the mean.

3. **Autocorrelation**:
   - High-order autocorrelation in spreads supports predictability, enhancing trading signal reliability.

### Performance Analysis:

1. **Base Strategy**:

   - Using a threshold of 1.5 standard deviations, 12 trades were executed from 2019 onward.
   - Wealth grew from $1,000 to $2,817, demonstrating profitability without leverage.

2. **Leveraged Trading**:

   - With 20x leverage, theoretical final wealth was $27,472, but bankruptcy occurred in 2021 due to large losses.

3. **Stop-Loss Implementation**:

   - Introduced a stop-loss at 2.75 standard deviations.
   - Increased trading frequency to 40 trades but reduced final wealth to $1,655, suggesting suboptimal results.

4. **Rolling Window Parameters**:

   - Rolling estimates introduced high volatility in spreads.
   - Resulted in frequent trades (377), reducing final wealth to $663.

5. **Cointegration Breakdown Protection**:
   - Closing positions during cointegration breakdown improved stability.
   - Final wealth was $1,302, with reduced downside risk.

---

## Conclusion

- **Profitability**:
  - Pair trading based on cointegration is viable but sensitive to parameter selection and market conditions.
- **Risk Management**:

  - Conservative strategies, such as cointegration breakdown protection, trade early gains for reduced risk.

- **Practical Considerations**:
  - Real-world implementation must account for trading fees, liquidity, and market impact, especially for institutional investors.

---

## Figures

Include charts for:

1. **Normalized Spread and Trading Signals**.
2. **Cumulative Wealth Over Time**.
3. **Impact of Stop-Loss and Rolling Window Estimates**.
4. **Performance with Cointegration Breakdown Protection**.
