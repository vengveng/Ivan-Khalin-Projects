import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
import itertools
import statsmodels.api as sm
from typing import Optional

# Functions

def simple_returns(series: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame: 
    return series.pct_change()

def log_returns(series: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    simple = simple_returns(series) # could get expensive
    return np.log1p(simple)

def annualized_mean(sample_mean: float, compounding_frequency: int) -> float:
    return (1 + sample_mean) ** compounding_frequency - 1

def annualized_variance(sample_variance: float, compounding_frequency: int) -> float:
    return sample_variance * compounding_frequency

def get_statistics(statistics: list, series: pd.Series | pd.DataFrame, annualized=False, compounding_frequency=365) -> pd.DataFrame:
    stats_list = [getattr(series, stat)() for stat in statistics]

    if annualized:
        for index, stat in enumerate(statistics):
            if stat == 'mean':
                stats_list[index] = annualized_mean(stats_list[index], compounding_frequency)
            if stat == 'var':
                stats_list[index] = annualized_variance(stats_list[index], compounding_frequency)

    return pd.DataFrame(stats_list, index=statistics)

def z_score(series: pd.Series, testValues):

    sample_mean = series.mean()
    sample_std = series.std() #ddof = 1
    return (testValues - sample_mean) / sample_std

def OLS(y: pd.Series, x:pd.Series):

    covXY = y.cov(x)
    varX = x.var()

    beta = covXY / varX
    alpha = y.mean() - beta * x.mean()

    residuals = y - (alpha + beta * x)
    SSE = np.sum(residuals**2)
    SE_beta = np.sqrt(SSE / (len(y) - 2) / np.sum((x - x.mean())**2))

    return alpha, beta, SE_beta

def ljung_box_test(data: pd.Series | np.ndarray, n_lags: int, confidence_level: float) -> dict:

    """
    Perform the Ljung-Box test for autocorrelation.
    """
    T = len(data)
    autocorrelations = [data.autocorr(lag=lag) for lag in range(1, n_lags + 1)]
    autocorrelations = np.array(autocorrelations)
    lags = np.arange(1, n_lags + 1)
    lbq_statistic = np.sum((autocorrelations ** 2) / (T - lags))
    lbq_statistic *= T * (T + 2)

    degrees_of_freedom = n_lags
    critical_value = chi2.ppf(confidence_level, degrees_of_freedom)
    pval = chi2.sf(lbq_statistic, degrees_of_freedom)

    if lbq_statistic > critical_value:
        decision = f"Evidence of autocorrelation at {confidence_level}"
    else:
        decision = f"No evidence of autocorrelation at {confidence_level}"

    result = {'lbq-statistic': lbq_statistic, 
              'critical-value': critical_value, 
              'verdict': decision,
              'p-value': pval
              }

    return result

def cointegration(aSeries: pd.Series, bSeries: pd.Series) -> float: # based on Philips and Ouliaris

    alpha, beta, SE = OLS(aSeries, bSeries)
    zhatt = aSeries - alpha - beta * bSeries 
    zhatt_1 = zhatt.shift(1)
    Dzhatt = zhatt - zhatt_1
    mu, phi, SE = OLS(Dzhatt[1:-1], zhatt_1[1:-1])
    return alpha, beta, phi / SE

# Filepaths for input data
root = 'EMF/project_1/'
dpPath = f'{root}data/Data_Project1.xlsx'

# Filepaths for output data - required for visualization
simPath = f'{root}data/simStatsDF.csv'
simVIIPath = f'{root}data/simStatsVII.csv'
simPOPath = f'{root}data/simStatsPO.csv'
ztPath = f'{root}data/ztTILDE.csv'
ptsPath = f'{root}data/ptStats.csv'
ptsl20Path = f'{root}data/ptStats_L20.csv'
ptsslPath = f'{root}data/ptStats_stoploss.csv'
ptsslisPath = f'{root}data/ptStats_stoploss_insample.csv'
rpPath = f'{root}data/rolling_parameters.csv'
ptsslispvlPath = f'{root}data/ptStats_stoploss_insample_pvaluelimit.csv'

# Loading data
column_names = ['HLT', 'MAR', 'IHG', 'BKNG', 'HYATT']
df = pd.read_excel(dpPath, sheet_name='Hotels', skiprows=1, index_col=0, parse_dates=True, names=column_names)
df.index.name = 'Date'

dailyData = df
weeklyData = df[df.index.dayofweek == 0]

daily_compounding_frequency = 365
weekly_compounding_frequency = 52
statistics = ['mean', 'var', 'skew', 'kurt', 'min', 'max']

# #-----------------------------------------------------------------------------#
#Q1.1
simpleDaily = simple_returns(dailyData)
logDaily = log_returns(dailyData)
sd_stats = get_statistics(statistics, simpleDaily, True, daily_compounding_frequency)
ld_stats = get_statistics(statistics, logDaily, True, daily_compounding_frequency)
print('simple daily:\n',sd_stats, '\n\n','simple log:\n', ld_stats, '\n\n\n')

# #-----------------------------------------------------------------------------#
#Q1.2
simpleWeekly = simple_returns(weeklyData)
logWeekly = log_returns(weeklyData)
sw_stats = get_statistics(statistics, simpleWeekly, True, weekly_compounding_frequency)
lw_stats = get_statistics(statistics, logWeekly, True, weekly_compounding_frequency)
# print('simple weekly:\n', sw_stats, '\n\n','log weekly:\n', lw_stats, '\n\n\n')

# #-----------------------------------------------------------------------------#
# #Q1.3
# # print(lw_stats - ld_stats, '\n\n\n')

# #-----------------------------------------------------------------------------#
#Q2.1
#Model 1
pt = np.log(dailyData['HLT'])
pt_1 = pt.shift(1)
alpha, beta, SE = OLS(pt[1:-1], pt_1[1:-1])
# print((beta-1)/SE) # ANSWER

rt = logDaily['HLT'] # log(P(t))-log(P(t-1)) = log(P(t)/P(t-1)) = log(1+R(t)) = r(t)
alpha, beta, SE = OLS(rt[1:-1], pt_1[1:-1])
# print((beta-0)/SE) # not -1, we are estimating gamma rather than phi # ANSWER

# #-----------------------------------------------------------------------------#
#Q2.4
T = dailyData.shape[0]
N = 10000
simStats = []

while N > 0:

    simErrors = np.random.normal(0, 1, T)
    simSeries = np.cumsum(simErrors) + np.log(1000) # Initial value to avoid negative logarithms
    pt = pd.Series(simSeries)
    pt_1 = pt.shift(1)
    mu, phi, SE = OLS(pt[1:-1], pt_1[1:-1])
    simStats.append((phi - 1) / SE)

    N += -1

simStatsDF = pd.Series(simStats) 
simStatsDF.to_csv(simPath)
statDictDF = {0.90: simStatsDF.quantile(0.10), 0.95: simStatsDF.quantile(0.05), 0.99: simStatsDF.quantile(0.01)}
# print(statDictDF)

# #-----------------------------------------------------------------------------#
#Q2.7
T = 500
N = 10000
simStatsVII = []

while N > 0:

    simErrors = np.random.normal(0, 1, T)
    simSeries = pd.Series(index=range(T),dtype='float64')
    simSeries[0] = np.log(1000)
    for t in range (1,T):
        simSeries[t] = 0.2 * simSeries[t-1] + simErrors[t]

    pt = simSeries
    pt_1 = pt.shift(1)
    mu, phi, SE = OLS(pt[1:-1], pt_1[1:-1])
    simStatsVII.append((phi - 1) / SE)

    N += -1

simStatsVII = pd.Series(simStatsVII) 
simStatsVII.to_csv(simVIIPath)
statDict = {0.90: simStatsVII.quantile(0.10), 0.95: simStatsVII.quantile(0.05), 0.99: simStatsVII.quantile(0.01)} # rename 
# print(statDict)

# #-----------------------------------------------------------------------------#
#Q2.8
simStatsDF.sort_values()
conf095 = statDictDF[0.95]
hotelStats = {}

for name, series in dailyData.items():
    pt = np.log(series)
    pt_1 = pt.shift(1)
    mu, phi, SE = OLS(pt[1:-1], pt_1[1:-1])
    t_stat = (phi-1)/SE
    p_value = np.sum(simStatsDF <= t_stat) / len(simStatsDF)
    hotelStats[name] = {'df-stat': t_stat.round(3), 'crit-val': conf095, 'p-val': p_value.round(3), 'rejected': t_stat<=conf095}

# print(hotelStats)
hotelResults = pd.DataFrame(hotelStats)
print(hotelResults)
  
# #-----------------------------------------------------------------------------# 
# Q3.1
T = 500
N = 10000
simStats = []

while N > 0:

    simErrorsA = np.random.normal(0, 1, T)
    simErrorsB = np.random.normal(0, 1, T)
    simSeriesA = pd.Series(simErrorsA).cumsum() + np.log(1000) # Initial value to avoid negative logarithms
    simSeriesB = pd.Series(simErrorsB).cumsum() + np.log(1000) # Initial value to avoid negative logarithms
    ptA = simSeriesA
    ptB = simSeriesB
    alpha, beta, SE = OLS(ptA, ptB)

    zhatt = ptA - alpha - beta * ptB
    zhatt_1 = zhatt.shift(1)
    Dzhatt = zhatt - zhatt_1
    alpha, beta, SE = OLS(Dzhatt[1:-1], zhatt_1[1:-1])

    simStats.append((beta) / SE)

    N += -1

simStatsPO = pd.Series(simStats)
simStatsPO.to_csv(simPOPath)
statDict = {0.90: simStatsPO.quantile(0.10), 0.95: simStatsPO.quantile(0.05), 0.99: simStatsPO.quantile(0.01)} # rename 
print(statDict)

# -----------------------------------------------------------------------------#
# Q3.2
simStatsPO = pd.read_csv(simPOPath, index_col=0).iloc[:, 0]
simStatsPO.sort_values()
simStatsPOlen = len(simStatsPO)

logPrices = np.log(dailyData)
asset_pairs = itertools.product(logPrices.columns, repeat=2)
cointegrationDictTSTAT = {name:{} for name in dailyData.columns}
cointegrationDictPVAL = {name:{} for name in dailyData.columns}
cointegrationDictALPHA = {name:{} for name in dailyData.columns}
cointegrationDictBETA = {name:{} for name in dailyData.columns}

for asset1, asset2 in asset_pairs:
    if asset1 == asset2:
        cointegrationDictTSTAT[asset1][asset2] = np.nan
        cointegrationDictPVAL[asset1][asset2] = np.nan
        cointegrationDictALPHA[asset1][asset2] = np.nan
        cointegrationDictBETA[asset1][asset2] = np.nan
    else:
        series1 = logPrices[asset1]
        series2 = logPrices[asset2]
        alpha, beta, t_stat = cointegration(series1, series2)
        p_value = np.sum(simStatsPO <= t_stat) / simStatsPOlen
        cointegrationDictTSTAT[asset1][asset2] = t_stat
        cointegrationDictPVAL[asset1][asset2] = p_value
        cointegrationDictALPHA[asset1][asset2] = alpha
        cointegrationDictBETA[asset1][asset2] = beta

# These parameters are the cointegration estimates, not the initital OLS
# The cointegration values are reported for Asset in column → Asset in row
cointegrationTSTAT = pd.DataFrame(cointegrationDictTSTAT)
cointegrationPVAL = pd.DataFrame(cointegrationDictPVAL)
cointegrationALPHA = pd.DataFrame(cointegrationDictALPHA)
cointegrationBETA = pd.DataFrame(cointegrationDictBETA)
statDict = {0.90: simStatsPO.quantile(0.10), 0.95: simStatsPO.quantile(0.05), 0.99: simStatsPO.quantile(0.01)}
print(cointegrationTSTAT, cointegrationPVAL, cointegrationALPHA, cointegrationBETA)
print(statDict)

#-----------------------------------------------------------------------------#
#Q4.1

PAt = dailyData['IHG']
PBt = dailyData['MAR']
alpha, beta, SE = OLS(PAt, PBt)
zt = PAt - alpha - beta * PBt
ztSD = zt.std(ddof=1)
ztTILDE = zt / ztSD # 
ztTILDE.to_csv(ztPath)

print(ljung_box_test(ztTILDE, 10, 0.95))

#-----------------------------------------------------------------------------#
# Q4.7
def trading_strategy(normalized_residual, threshold_divergence, assetA_price, assetB_price, 
                     monetary_wealth, leverage, alpha, beta, previous_position, position_flag, stopLoss):
    
    '''
    This function takes our current wealth and the market conditions. It returns us the position governed 
    by our trading strategy and a flag indicating which position is currently active. The flag is needed
    to tell the loop within which this function is contained about which parameters must be updated.
    '''

    PA = assetA_price
    PB = assetB_price
    W = monetary_wealth
    L = leverage
    Q1 = np.array((-1, beta))
    Q2 = np.array((1, -beta))
    stoploss = stopLoss

    if alpha > 0:
        if normalized_residual > threshold_divergence:
            Q = L * W / PA * Q1
            position_flag = 'Q1'
            if normalized_residual > stoploss:
                Q = np.array((0, 0))
                position_flag = 'Q0'

        elif normalized_residual < -threshold_divergence:
            Q = L * W / (beta * PB + L * (PA - beta * PB)) * Q2
            position_flag = 'Q2'
            if normalized_residual < -stoploss:
                Q = np.array((0, 0))
                position_flag = 'Q0'

        elif position_flag == 'Q1' and normalized_residual <= 0:
            Q = np.array((0, 0))
            position_flag = 'Q0'

        elif position_flag == 'Q2' and normalized_residual >= 0:
            Q = np.array((0, 0))
            position_flag = 'Q0'
        else:
            Q = previous_position

    # Our alpha is positive so this doesn't get used
    if alpha < 0:
        if normalized_residual > threshold_divergence:
            Q = L * W / (PA - L * (PA - beta * PB)) * Q1
            position_flag = 'Q1'
            if normalized_residual > stoploss:
                Q = np.array((0, 0))
                position_flag = 'Q0'            

        elif normalized_residual < -threshold_divergence:
            Q = L * W / (beta * PA) * Q2
            position_flag = 'Q2'
            if normalized_residual < -stoploss:
                Q = np.array((0, 0))
                position_flag = 'Q0'            

        elif position_flag == 'Q1' and normalized_residual <= 0:
            Q = np.array((0, 0))
            position_flag = 'Q0'

        elif position_flag == 'Q2' and normalized_residual >= 0:
            Q = np.array((0, 0))
            position_flag = 'Q0'
        else:
            Q = previous_position

    return Q, position_flag

def daily_trade(analysis_index: pd.DatetimeIndex, normalized_spread: pd.Series, pValue: Optional[pd.Series]=None) -> pd.DataFrame:
    
    columns=['equity_wealth', 'position_change', 'leverage']
    PT_data = pd.DataFrame(index=analysis_index, columns=columns, dtype=float)

    simpleDaily.iloc[0] = 0

    for day in analysis_index:

        # Assigning function inputs
        PA = dailyData['IHG'][day]
        PB = dailyData['MAR'][day]
        previousPosition = tradingParameters['previous_position']
        previousFlag = tradingParameters['position_flag']
        tradingParameters['normalized_residual'] = normalized_spread[day]
        tradingParameters['assetA_price'] = PA
        tradingParameters['assetB_price'] = PB
        leverageLimit = tradingParameters['leverage']
        
        # Determining strategy
        positionQuantity, positionFlag = trading_strategy(**tradingParameters)
        tradingParameters['position_flag'] = positionFlag

        # Cointegration breakdown safeguard
        if pValue is not None and pValue[day] > 0.15:
            positionFlag = 'Q0'
            tradingParameters['position_flag'] = positionFlag
            positionQuantity = np.array((0, 0))

        # We must evaluate our leverage to not surpass the limit
        if positionFlag == 'Q1':
            currentLeverage = PA * -previousPosition[0] / tradingParameters['monetary_wealth']
        elif positionFlag == 'Q2':
            currentLeverage = PB * -previousPosition[1] / tradingParameters['monetary_wealth']
        else:
            currentLeverage = 0

        # If our position changes, than we spend money to buy in, but there are no capital gains for that period.
        # Our position changes if either the signal has changed, or we hit the leverage limit.
        if positionFlag != previousFlag or currentLeverage >= leverageLimit:
            positionChange = positionQuantity - previousPosition
            wealthChange = np.array((PA, PB)) @ positionChange
            tradingParameters['monetary_wealth'] -= wealthChange
            tradingParameters['previous_position'] = positionQuantity
            PT_data['position_change'][day] = 1

        # If our position does not change, our monetary wealth does not change, but there are capital gains for that period.
        elif positionFlag == previousFlag:
            tradingParameters['previous_position'] = previousPosition
            PT_data['position_change'][day] = 0

        # Storing for visualization
        equityWealth = tradingParameters['monetary_wealth'] + tradingParameters['previous_position'] @ np.array((PA, PB))

        PT_data['equity_wealth'][day] = equityWealth
        PT_data['leverage'][day] = currentLeverage

    return PT_data

tradingParameters = {'monetary_wealth': 1000.0,
                     'alpha': alpha,
                     'beta': beta,
                     'leverage': 2,
                     'normalized_residual': None,
                     'threshold_divergence': 1.5,
                     'assetA_price': None,
                     'assetB_price': None,
                     'previous_position': np.array((0, 0)),
                     'position_flag': 'Q0',
                     'stopLoss': np.inf
                     }

PT_data = daily_trade(dailyData.index, ztTILDE)
PT_data.to_csv(ptsPath)


#-----------------------------------------------------------------------------#
# Q4.8
tradingParameters = {'monetary_wealth': 1000.0,
                     'alpha': alpha,
                     'beta': beta,
                     'leverage': 20,
                     'normalized_residual': None,
                     'threshold_divergence': 1.5,
                     'assetA_price': None,
                     'assetB_price': None,
                     'previous_position': np.array((0, 0)),
                     'position_flag': 'Q0',
                     'stopLoss': np.inf
                     }

PT_data = daily_trade(dailyData.index, ztTILDE)
PT_data.to_csv(ptsl20Path)

#-----------------------------------------------------------------------------#
#Q4.9
z_in = 1.5
z_stop = 1.75
y = ztTILDE[1:-1]
x = ztTILDE.shift(1)[1:-1]
phi_0, phi_1, _ = OLS(y, x)
epsilon = y - (phi_0 + phi_1 * x)
epsilonSD = epsilon.std(ddof=0)
z_hat = phi_0 + phi_1 * z_in

print(1 - norm.cdf(z_stop, z_hat, epsilonSD))

#-----------------------------------------------------------------------------#
#4.10
tradingParameters = {'monetary_wealth': 1000.0,
                     'alpha': alpha,
                     'beta': beta,
                     'leverage': 2,
                     'normalized_residual': None,
                     'threshold_divergence': 1.5,
                     'assetA_price': None,
                     'assetB_price': None,
                     'previous_position': np.array((0, 0)),
                     'position_flag': 'Q0',
                     'stopLoss': 2.75
                     }

PT_data = daily_trade(dailyData.index, ztTILDE)
PT_data.to_csv(ptsslPath)

#-----------------------------------------------------------------------------#
#4.11, 4.12

simStatsPO = pd.read_csv(simPOPath, index_col=0).iloc[:, 0]
simStatsPO.sort_values()
simStatsPOlen = simStatsPO.shape[0]

signals = pd.Series(dtype=float, index=dailyData.index)
columns = ['return_correlation', 'price_correlation', 'alpha', 'alpha_full', 'beta', 'beta_full', 'ztilde', 'ztilde_full', 'CI_pval', 'CI_tstat']
parameters = pd.DataFrame(index=dailyData.index, columns=columns, dtype=float)

for start in range(0, dailyData.shape[0], 20):
    end = start + 500
    window_data = dailyData.iloc[start:end]

    # Parameter estimation
    PAt = window_data['IHG']
    PBt = window_data['MAR']
    alpha, beta, SE = OLS(PAt, PBt)
    zt = PAt - alpha - beta * PBt
    ztSD = zt.std(ddof=1)
    ztTILDE = zt / ztSD 
    parameters['alpha'].iloc[start:end] = alpha
    parameters['beta'].iloc[start:end] = beta

    # Correlation calculation
    window_data = simpleDaily.iloc[start:end]
    RAt = window_data['IHG']
    RBt = window_data['MAR']
    return_correlation = RAt.corr(RBt)
    price_correlation = PAt.corr(PBt)
    parameters['return_correlation'].iloc[start:end] = return_correlation
    parameters['price_correlation'].iloc[start:end] = price_correlation
    
    next_end = min(end + 20, dailyData.shape[0])
    next_window_data = dailyData.iloc[end:next_end]
    if not next_window_data.empty:
        zt = next_window_data['IHG'] - (alpha + beta * next_window_data['MAR'])
        # Standard deviation of z_t during the estimation period
        ztSD = zt.std(ddof=1)
        # Normalized signal z_tilde
        ztTILDE = zt / ztSD
        signals.iloc[end:next_end] = ztTILDE
        parameters['ztilde'].iloc[end:next_end] = ztTILDE

    # Testing for cointegration for Q4.14 here
    logPAt = np.log(PAt)
    logPBt = np.log(PBt)
    alpha, beta, t_stat = cointegration(PAt, PBt)
    p_value = np.sum(simStatsPO <= t_stat) / simStatsPOlen
    parameters['CI_pval'].iloc[start:end] = p_value
    parameters['CI_tstat'].iloc[start:end] = t_stat
    
# Computing full sample performance once more
ztTILDE_rolling = signals

PAt = dailyData['IHG']
PBt = dailyData['MAR']
alpha, beta, SE = OLS(PAt, PBt)
zt = PAt - alpha - beta * PBt
ztSD = zt.std(ddof=1)
ztTILDE = zt / ztSD 

parameters['alpha_full'] = alpha
parameters['beta_full'] = beta
parameters['ztilde_full'] = ztTILDE

# The results of the rolling window and the full sample estimation stored here
parameters.to_csv(rpPath)

#-----------------------------------------------------------------------------#
#Q4.13

tradingParameters = {'monetary_wealth': 1000.0,
                     'alpha': alpha,
                     'beta': beta,
                     'leverage': 2,
                     'normalized_residual': None,
                     'threshold_divergence': 1.5,
                     'assetA_price': None,
                     'assetB_price': None,
                     'previous_position': np.array((0, 0)),
                     'position_flag': 'Q0',
                     'stopLoss': 2.75
                     }

PT_data = daily_trade(dailyData.loc['2015-12-02':].index, ztTILDE_rolling)
PT_data.to_csv(ptsslisPath)

#-----------------------------------------------------------------------------#
#Q4.15

pValues = parameters['CI_pval']

tradingParameters = {'monetary_wealth': 1000.0,
                     'alpha': alpha,
                     'beta': beta,
                     'leverage': 2,
                     'normalized_residual': None,
                     'threshold_divergence': 1.5,
                     'assetA_price': None,
                     'assetB_price': None,
                     'previous_position': np.array((0, 0)),
                     'position_flag': 'Q0',
                     'stopLoss': 2.75
                     }

PT_data = daily_trade(dailyData.loc['2015-12-02':].index, ztTILDE_rolling, pValues)
PT_data.to_csv(ptsslispvlPath)