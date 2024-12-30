# Data Science Project: Impact of Strikes on Firm Stock Returns

### Group Members:

- Arnaud Cavallin
- Weilin Wang
- Eva Errecart
- Fraser Levick
- Joseph de Preux
- Julien Bovet
- Ivan Khalin
- Jeremy Bourqui
- Litian Zhang
- Svetlana Sokovykh
- Matthieu Meuriot

---

## Abstract

This research investigates the effects of labor strikes on daily stock returns of firms using panel regressions on U.S. data from 1984 to 2020. The findings suggest that strikes and the proportion of strikers positively impact stock returns, a result attributed to market efficiency and filtered data considerations. Industry-specific variations are also highlighted.

---

## 1. Introduction & Economic Context

Strikes are critical intersections between labor and capital, affecting firms' market performance. This study examines how strikes influence daily stock returns, providing updated insights into labor disruption effects on financial markets. Past research has become outdated, focusing on older data, which this paper aims to address with recent observations.

---

## 2. Literature Review

Key findings from past research:

- **Strikes' Economic Costs**: Significant costs vary across industries but have declined over time (Olson et al., 1986).
- **Duration's Role**: Longer strikes can have positive effects on returns after resolution (Nelson et al., 2002).
- **Market Efficiency**: Negative effects are priced in before strike onset (Bhana, 1997).
- **Industry & Market Dynamics**: Impact varies by market conditions and industry characteristics (Yip et al., 2007).

---

## 3. Methodology

### 3.1 Data Collection

- Data retrieved from the Federal Mediation and Conciliation Service (FMCS) on over 14,000 U.S. strikes.
- Challenges included cleaning inconsistent company names and matching them to ISIN codes using the `FuzzyWuzzy` library.
- Additional data: S&P 500 index, company-specific metrics (e.g., employees, debt-to-equity ratio).

### 3.2 Panel Regression Models

#### Naïve Panel Regression

- Used unfiltered data with firm-level and time-specific variables.
- Results were inconclusive due to noise from non-strike periods.

#### Filtered Panel Regression

- Focused on strike event windows (one month before, during, and after strikes).
- Incorporated firm fixed effects, improving statistical significance.

---

## 4. Results

### Key Findings:

1. **Overall Impact**:

   - Strikes and the proportion of strikers positively affect stock returns during strike periods.
   - Results align with market efficiency theory: pre-strike price adjustments anticipate disruptions.

2. **Industry-Specific Effects**:

   - Utilities and Manufacturing: Positive impact of strikes on stock returns.
   - Construction: Negative impact due to operational halts.
   - Retail: Negative correlation with striker proportion.

3. **Interaction Terms**:
   - Rolling average effects suggest mean reversion trends.
   - Interaction between S&P 500 returns and strike parameters highlights index influence on firm returns.

---

## 5. Conclusion

This study supports previous findings that strikes influence stock returns and highlights the importance of market efficiency in pricing disruptions. Future research should:

1. Focus on industry-specific analyses for more granular insights.
2. Explore additional interaction terms to capture complex market dynamics.

---

## Figures and Tables

Include:

1. **Panel Regression Results by Industry**.
2. **Impact of Strike Duration on Returns**.
3. **Correlation Between Proportionate Strikers and Returns**.
