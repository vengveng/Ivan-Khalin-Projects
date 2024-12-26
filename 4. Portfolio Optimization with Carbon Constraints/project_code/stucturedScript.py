import pandas as pd 
import numpy as np
import cvxpy as cp

def excel_loader(path):
    global theta_scale
    theta_scale = 1
    """
    Load and preprocess Excel data.

    This function is specifically designed for loading and preprocessing data for the scope.

    Args:
        path (str): The path to the Excel file.

    Returns:
        pandas.DataFrame: The preprocessed data.

    """
    data = pd.read_excel(path, usecols=lambda x: x != 'NAME', index_col=0).transpose()
    data = data[filterEM]
    data.index = pd.to_datetime(data.index, format='%Y')
    data.index = data.index + pd.offsets.YearEnd()
    data.index.rename('DATE', inplace=True)
    data = data[data.index.year > 2004]
    nan_columns = data.iloc[0].loc[data.iloc[0].isna()].index
    data.loc['2005-12-31', nan_columns] = data.loc['2006-12-31', nan_columns]
    data.interpolate(method='linear', axis=0, inplace=True)

    return data

def annualized_mean(sample_mean: float) -> float:
    return (1 + sample_mean) ** 12 - 1

def annualized_volatility(sample_std: float) -> float:
    return sample_std * np.sqrt(12)

def sharpe_ratio(mean: float, volatility: float) -> float:
    return mean / volatility

def expected_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return returns.mean(axis=0)

def expected_covariance(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    dimension = returns.shape[0]
    MU = expected_returns(returns)
    ER = returns - MU
    return ER.T @ ER / (dimension)

def portfolio_evaluation(monthlyReturns: pd.Series | np.ndarray, monthlyRFrate: pd.Series) -> dict:

    '''
    Evaluates the performance of a portfolio given its monthly returns. 
    It calculates and returns a dictionary containing the annualized mean return,
    annualized volatility, Sharpe ratio, minimum return, and maximum return of the portfolio.
    monthlyRFrate must be indexed by and be of same length as the sample of monthly returns 
    that is being evaluated.
    '''

    mean = monthlyReturns.mean()
    volatility = monthlyReturns.std()
    annualizedMean = annualized_mean(mean)
    annualizedVolatility = annualized_volatility(volatility)
    monthlyExcessReturn = monthlyReturns.sub(monthlyRFrate, axis=0)
    meanExcessReturn = monthlyExcessReturn.mean()
    annualizedExcessReturn = annualized_mean(meanExcessReturn)
    sharpeRatio = sharpe_ratio(annualizedExcessReturn, annualizedVolatility)
    minimum = monthlyReturns.min()
    maximum = monthlyReturns.max()

    portfolio_performance = {
        'mu': annualizedMean,
        'std': annualizedVolatility,
        'SR': sharpeRatio,
        'min': minimum,
        'max': maximum
    }

    return portfolio_performance

def save_portfolio_data(portfolio_performance, portfolio_values, portfolio_returns, monthly_weights, carbon_footprint, portfolio_waci, name=None):
    valid_names = ['SAP', 'SAP_50R', 'BP', 'BP_50R', 'BP_TENZR']
    if name is None:
        raise KeyError('Please provide a name for the portfolio data.')
    if name not in valid_names:
        raise ValueError(f'Invalid name. Please choose one of the following: {valid_names}')
    
    path = savePaths[name]   
     
    portfolio_performance = pd.DataFrame(portfolio_performance)
    monthly_weights = monthly_weights[monthly_weights.index.year > 2005]
    portfolio_performance.to_csv(path['stats'])
    monthly_weights.to_csv(path['weights'])
    portfolio_returns.to_csv(path['returns'])
    portfolio_values.to_csv(path['value'])
    carbon_footprint.to_csv(path['CF'])
    portfolio_waci.to_csv(path['WACI'])

def reset_portfolio_data():
    global portfolio_value, portfolio_values, portfolio_CF, portfolio_performance, portfolio_returns, monthly_weights, portfolio_waci
    portfolio_value = 1e6
    portfolio_values = pd.Series(dtype=float) # For collecting annual portfolio values
    portfolio_CF = pd.Series(dtype=float) # For collecting annual carbon footprint
    portfolio_performance = {} # For collecting annual portfolio statistics
    portfolio_returns = pd.Series(dtype=float) # For collecting monthly returns
    portfolio_waci = pd.Series(dtype=float) # For collecting monthly WACI values
    monthly_weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    monthly_weights.fillna(0, inplace=True)

# Filepaths for input data
staticPath = 'data/Static.xlsx'
ritPath = 'data/DS_RI_T_USD_M.xlsx'
rfPath = 'data/Risk_Free_Rate.xlsx'
mvPath = 'data/DS_MV_USD_M.xlsx'
scope1Path = 'data/TC_Scope1.xlsx'  
scope2Path = 'data/TC_Scope2.xlsx'
scope3Path = 'data/TC_Scope3.xlsx'
scope1IntensityPath = 'data/TC_Scope1Intensity.xlsx'
scope2IntensityPath = 'data/TC_Scope2Intensity.xlsx'
scope3IntensityPath = 'data/TC_Scope3Intensity.xlsx'

# Filepaths for output data - required for visualization - do not need to be modified
savePaths = {
    'SAP': {
        'weights': 'data/visualization/SAP_monthly_weights.csv',
        'stats': 'data/visualization/SAP_stats.csv',
        'returns': 'data/visualization/SAP_monthly_returns.csv',
        'value': 'data/visualization/SAP_portfolio_value.csv',
        'CF': 'data/visualization/SAP_carbonFootprint.csv',
        'WACI': 'data/visualization/SAP_WACI.csv'
    },
    'SAP_50R': {
        'weights': 'data/visualization/SAP_50R_monthly_weights.csv',
        'stats': 'data/visualization/SAP_50R_stats.csv',
        'returns': 'data/visualization/SAP_50R_monthly_returns.csv',
        'value': 'data/visualization/SAP_50R_portfolio_value.csv',
        'CF': 'data/visualization/SAP_50R_carbonFootprint.csv',
        'WACI': 'data/visualization/SAP_50R_WACI.csv'
    },
    'BP': {
        'weights': 'data/visualization/BP_monthly_weights.csv',
        'stats': 'data/visualization/BP_stats.csv',
        'returns': 'data/visualization/BP_monthly_returns.csv',
        'value': 'data/visualization/BP_portfolio_value.csv',
        'CF': 'data/visualization/BP_carbonFootprint.csv',
        'WACI': 'data/visualization/BP_WACI.csv'
    },
    'BP_50R': {
        'weights': 'data/visualization/BP_50R_monthly_weights.csv',
        'stats': 'data/visualization/BP_50R_stats.csv',
        'returns': 'data/visualization/BP_50R_monthly_returns.csv',
        'value': 'data/visualization/BP_50R_portfolio_value.csv',
        'CF': 'data/visualization/BP_50R_carbonFootprint.csv',
        'WACI': 'data/visualization/BP_50R_WACI.csv'
    },
    'BP_TENZR': {
        'weights': 'data/visualization/BP_TENZR_monthly_weights.csv',
        'stats': 'data/visualization/BP_TENZR_stats.csv',
        'returns': 'data/visualization/BP_TENZR_monthly_returns.csv',
        'value': 'data/visualization/BP_TENZR_portfolio_value.csv',
        'CF': 'data/visualization/BP_TENZR_carbonFootprint.csv',
        'WACI': 'data/visualization/BP_TENZR_WACI.csv'
    }
}

# Create EM filter
staticData = pd.read_excel(staticPath, engine='openpyxl')
filterEM = staticData['ISIN'][staticData['Region'] == 'EM']

# Load Data
masterData = pd.read_excel(ritPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
masterData = masterData[filterEM]
masterData.index.rename('DATE', inplace=True) # print(sum(masterData.isna().any())) # Prices have no missing values

rfRate = pd.read_excel(rfPath, index_col=0, engine='openpyxl')
rfRate = rfRate.iloc[:masterData.shape[0]]
rfRate.set_index(masterData.index, inplace=True)
rfRate = rfRate.squeeze() / 100

scope1 = excel_loader(scope1Path)
scope2 = excel_loader(scope2Path)
scope3 = excel_loader(scope3Path)
scope1Intensity = excel_loader(scope1IntensityPath)
scope2Intensity = excel_loader(scope2IntensityPath)
scope3Intensity = excel_loader(scope3IntensityPath)
emissions = scope1 + scope2 + scope3
intensity = scope1Intensity + scope2Intensity + scope3Intensity

capData = pd.read_excel(mvPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
capData = capData[filterEM]
capData.index = pd.to_datetime(capData.index, format='%Y-%m-%d')
capData.index.rename('DATE', inplace=True)

# Prepare Data
returns = masterData.pct_change()
returns.loc['2000-01-31'] = returns.loc['2000-02-29'] # Backfill first observation
returns.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
returns.fillna(0, inplace=True)

# Iteration Index Creation
'''
Every time an allocation is made, there is a sample of data that is used for min-variance optimization,
and there is a sample of data used for evaluating performance. Here, we create a dictionary of time indexes which
will pass the appropriate date ranges (indexes) to the iterating function. The dictionary is initialized for the
date ranges corresponding to the original 'in sample' data. A loop is used to itterate through the subsequent 
sets of indexes. Importantly, the enitre set of data up to the begining of the sample is taken for optimization (this
was changed by the addition of the second condition "& (masterData.index.year >= 2000 + index)").
'''

def iteration_depth(limit=None):
    if limit is None:
        YYYY = 2021
    else:
        YYYY = limit
    indexIterator = {0: {'optimizationIndex': masterData.index.year < 2006, 'evaluationIndex': masterData.index.year == 2006}}
    for year, index in zip(range(2007, YYYY + 1), range(1, 22 + 1)):
        optimizationIndex = (masterData.index.year < year) & (masterData.index.year >= 2000 + index)
        evaluationIndex = masterData.index.year == year
        indexIterator[index] = {'optimizationIndex': optimizationIndex, 'evaluationIndex': evaluationIndex}
    return indexIterator

#------------------------------------------------------------#
# 1.1 - Standard Asset Allocation
print('1.1 - Standard Asset Allocation')
indexIterator = iteration_depth(2021)
reset_portfolio_data()

def create_filter_mask(sampleData):
    
    highestYearEnd = sampleData.index.max()
    highestYearStart = pd.to_datetime(f'{highestYearEnd.year}-01-31')
    
    # Zero December Returns
    decemberData = sampleData.loc[[highestYearEnd]]
    decemberFilter = decemberData.columns[decemberData.iloc[0] == np.inf] # deactivated

    # December price below threshold
    yearEndPrices = masterData.loc[highestYearEnd]
    priceFilter = yearEndPrices[yearEndPrices < -np.inf].index # activated

    # High return filter
    returnFilterHigh = sampleData.columns[sampleData.max() >= np.inf] # deactivated
    returnFilterLow = sampleData.columns[sampleData.min() <= -np.inf] # deactivated
    returnFilter = returnFilterHigh.union(returnFilterLow)
    
    # Frequent Zero Returns
    yearlyData = sampleData.loc[highestYearStart:highestYearEnd]
    monthsWithZeroReturns = (yearlyData == 0).sum(axis=0)
    frequentZerosFilter = monthsWithZeroReturns[monthsWithZeroReturns >= 12].index # activated

    return decemberFilter.union(frequentZerosFilter).union(priceFilter).union(returnFilter)

for step in indexIterator:

    # Loading Data
    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    sampleData = returns[optimizationIndex]
    evaluationData = returns[evaluationIndex]
    evaluationRF = rfRate[evaluationIndex]

    currentYear = pd.to_datetime(f'{step + 2006}-12-31')

    nullFilter = create_filter_mask(sampleData)
    sampleData = sampleData.drop(columns=nullFilter)
    evaluationData = evaluationData.drop(columns=nullFilter)
    evaluationEmissions = emissions.drop(columns=nullFilter)
    evaluationIntensity = intensity.drop(columns=nullFilter)
    evaluationCAP = capData.drop(columns=nullFilter)
    print(evaluationData.shape)
    
    # Optimization
    #------------------------------------------------------------#
    N = sampleData.shape[1]
    weights = cp.Variable(N)
    covarianceMatrix = expected_covariance(sampleData)
    portfolio_variance = cp.quad_form(weights, covarianceMatrix)
    objective = cp.Minimize(portfolio_variance)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    portfolioWeights = weights.value
    print(f"{step}: Status: {problem.status}, Objective value: {problem.value}")
    #------------------------------------------------------------#

    # Portfolio Performance Evaluation Section
    # Monthly Weights Adjustment
    monthlyReturns = []
    monthlyWeights = [portfolioWeights]
    for singleMonthReturns in evaluationData.values:
        portfolioReturns = monthlyWeights[-1] @ singleMonthReturns * theta_scale
        portfolioWeights = monthlyWeights[-1] * (1 + singleMonthReturns) / (1 + portfolioReturns)
        monthlyReturns.append(portfolioReturns)
        monthlyWeights.append(portfolioWeights)
        
    monthlyReturns = pd.Series(monthlyReturns, index=evaluationData.index)

    # Portfolio Value Calculation
    portfolio_value = portfolio_value * (1 + np.array(monthlyReturns)).prod()
    portfolio_values[currentYear] = portfolio_value

    # Portfolio Carbon Footprint Calculation
    meanMonthlyWeights = np.mean(monthlyWeights, axis=0)
    USD_firmValue = meanMonthlyWeights * portfolio_value
    ownership = USD_firmValue / (evaluationCAP.loc[currentYear])
    emissions_xxx = np.array(evaluationEmissions.loc[currentYear])
    carbonFootprint = 1 / portfolio_value * np.sum(ownership * emissions_xxx)
    portfolio_CF[currentYear] = carbonFootprint

    # WACI Calculation
    portfolio_waci[currentYear] = meanMonthlyWeights @ evaluationIntensity.loc[currentYear]

    # Tracking Other Metrics
    portfolio_performance[2006 + step] = portfolio_evaluation(monthlyReturns, evaluationRF)
    portfolio_returns = pd.concat([portfolio_returns, monthlyReturns])
    monthlyWeights = pd.DataFrame(monthlyWeights[1:], index=evaluationData.index, columns=evaluationData.columns)
    monthly_weights.loc[evaluationIndex, monthlyWeights.columns] = monthlyWeights.values

    print('CF:', carbonFootprint)
    print('PV:', portfolio_value)
    print(step, monthlyReturns)

# CSV
save_portfolio_data(portfolio_performance, portfolio_values, portfolio_returns, monthly_weights, portfolio_CF, portfolio_waci, name='SAP')

#------------------------------------------------------------#
# 1.2 - Benchmark Portfolio
reset_portfolio_data()
# Loading Market Value Data
capData = pd.read_excel(mvPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
capData = capData[filterEM]
capData.index.rename('DATE', inplace=True)

# Portfolio Weights and Returns
capitalization = capData.sum(axis=1)
annual_capitalization = capitalization.resample('Y').last()
benchmark_weights = capData.div(capitalization, axis=0).shift(1)
monthlyReturns = (benchmark_weights.mul(returns, axis=0)).sum(axis=1)
portfolio_performance = portfolio_evaluation(monthlyReturns, rfRate)
portfolio_performance = pd.DataFrame(portfolio_performance, index=['2000-2022']).T
portfolio_values = (1 + monthlyReturns).cumprod()
portfolio_values = portfolio_values / portfolio_values.loc['2006-01-31'] * 1e6

benchmark_CF = pd.Series(dtype=float)
for i in emissions.index:
    USD_firmValue = benchmark_weights.loc[i] * portfolio_values.loc[i]
    ownership = USD_firmValue / (capData.loc[i])
    emissions_xxx = np.array(emissions.loc[i])
    carbonFootprint = 1 / portfolio_values.loc[i] * np.sum(ownership * emissions_xxx)
    benchmark_CF[i] = carbonFootprint

annual_benchmark_weights = benchmark_weights.resample('Y').last()
portfolio_waci = (annual_benchmark_weights * intensity).sum(axis=1)

# CSV
save_portfolio_data(portfolio_performance, portfolio_values, monthlyReturns, benchmark_weights, benchmark_CF, portfolio_waci, name='BP')

#------------------------------------------------------------#
# 2.1 - Allocation with a 50% Reduction in Carbon Emissions
print('2.1 - Allocation with a 50% Reduction in Carbon Emissions')
target_CF = pd.read_csv(savePaths['SAP']['CF'], index_col=0, parse_dates=True)
target_CF.loc[pd.to_datetime('2005-12-31')] = target_CF.loc['2006-12-31']
target_CF.sort_index(inplace=True)
# indexIterator = iteration_depth(2006)
reset_portfolio_data()

for step in indexIterator:

    # Loading Data
    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    sampleData = returns[optimizationIndex]
    evaluationData = returns[evaluationIndex]
    evaluationRF = rfRate[evaluationIndex]

    currentYear = pd.to_datetime(f'{step + 2006}-12-31')
    previousYear = pd.to_datetime(f'{step + 2005}-12-31')

    nullFilter = create_filter_mask(sampleData)
    sampleData = sampleData.drop(columns=nullFilter)
    evaluationData = evaluationData.drop(columns=nullFilter)
    evaluationEmissions = emissions.drop(columns=nullFilter)
    evaluationIntensity = intensity.drop(columns=nullFilter)
    evaluationCAP = capData.drop(columns=nullFilter)
    print(evaluationData.shape)

    N = sampleData.shape[1]
    weights = cp.Variable(N)
    covarianceMatrix = expected_covariance(sampleData)
    cap_values = np.array(evaluationCAP.loc[previousYear])
    emissions_values = np.array(evaluationEmissions.loc[previousYear])

    USD_firm_value = cp.multiply(weights, portfolio_value)
    ownership = cp.multiply(USD_firm_value, 1 / cap_values)
    total_emissions = cp.sum(cp.multiply(ownership, emissions_values))
    carbon_footprint_constraint = cp.multiply(total_emissions, 1 / portfolio_value) <= 0.5 * target_CF.loc[previousYear]

    portfolio_variance = cp.quad_form(weights, covarianceMatrix)
    objective = cp.Minimize(portfolio_variance)
    constraints = [cp.sum(weights) == 1, weights >= 0, carbon_footprint_constraint]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    portfolioWeights = weights.value
    print(f"Step {step}: Status: {problem.status}, Objective value: {problem.value}")

    # Portfolio Performance Evaluation Section
    # Monthly Weights Adjustment
    monthlyReturns = []
    monthlyWeights = [portfolioWeights]
    for singleMonthReturns in evaluationData.values:
        portfolioReturns = monthlyWeights[-1] @ singleMonthReturns
        portfolioWeights = monthlyWeights[-1] * (1 + singleMonthReturns) / (1 + portfolioReturns)
        monthlyReturns.append(portfolioReturns)
        monthlyWeights.append(portfolioWeights)
        
    monthlyReturns = pd.Series(monthlyReturns, index=evaluationData.index)

    # Portfolio Value Calculation
    portfolio_value = portfolio_value * (1 + np.array(monthlyReturns)).prod()
    portfolio_values[currentYear] = portfolio_value

    # Portfolio Carbon Footprint Calculation
    meanMonthlyWeights = np.mean(monthlyWeights, axis=0)
    USD_firmValue = meanMonthlyWeights * portfolio_value
    ownership = USD_firmValue / (evaluationCAP.loc[currentYear])
    emissions_xxx = np.array(evaluationEmissions.loc[currentYear])
    carbonFootprint = 1 / portfolio_value * np.sum(ownership * emissions_xxx)
    portfolio_CF[currentYear] = carbonFootprint

    # WACI Calculation
    portfolio_waci[currentYear] = meanMonthlyWeights @ evaluationIntensity.loc[currentYear]

    # Tracking Other Metrics
    portfolio_performance[2006 + step] = portfolio_evaluation(monthlyReturns, evaluationRF)
    portfolio_returns = pd.concat([portfolio_returns, monthlyReturns])
    monthlyWeights = pd.DataFrame(monthlyWeights[1:], index=evaluationData.index, columns=evaluationData.columns)
    monthly_weights.loc[evaluationIndex, monthlyWeights.columns] = monthlyWeights.values

    print('CF:', carbonFootprint)
    print('PV:', portfolio_value)
    print(step, monthlyReturns)

# CSV
save_portfolio_data(portfolio_performance, portfolio_values, portfolio_returns, monthly_weights, portfolio_CF, portfolio_waci, name='SAP_50R')

#------------------------------------------------------------#

# 2.3 - Tracking Error Minimization
print('2.3 - Tracking Error Minimization')
target_CF = benchmark_CF
# indexIterator = iteration_depth(2021)
reset_portfolio_data()

for step in indexIterator:

    # Loading Data
    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    sampleData = returns[optimizationIndex]
    evaluationData = returns[evaluationIndex]
    evaluationRF = rfRate[evaluationIndex]

    nullFilter = create_filter_mask(sampleData)
    sampleData = sampleData.drop(columns=nullFilter)
    evaluationData = evaluationData.drop(columns=nullFilter)
    evaluationEmissions = emissions.drop(columns=nullFilter)
    evaluationIntensity = intensity.drop(columns=nullFilter)
    evaluationCAP = capData.drop(columns=nullFilter)
    print(evaluationData.shape)

    currentYear = pd.to_datetime(f'{step + 2006}-12-31')
    previousYear = pd.to_datetime(f'{step + 2005}-12-31')

    # Optimization
    N = sampleData.shape[1]
    weights = cp.Variable(N)
    covarianceMatrix = expected_covariance(sampleData)
    cap_values = np.array(evaluationCAP.loc[previousYear])
    emissions_values = np.array(evaluationEmissions.loc[previousYear])

    clean_benchmark = benchmark_weights.drop(columns=nullFilter)
    benchmark_alpha_Y = np.array(clean_benchmark.loc[previousYear])

    # USD_firm_value = weights * portfolio_value
    USD_firm_value = cp.multiply(weights, portfolio_value)
    ownership = cp.multiply(USD_firm_value, 1 / cap_values)
    total_emissions = cp.sum(cp.multiply(ownership, emissions_values))
    carbon_footprint_constraint = total_emissions / portfolio_value <= 0.5 * target_CF.loc[previousYear]

    portfolio_variance = cp.quad_form(1e3 * (weights - benchmark_alpha_Y), covarianceMatrix)
    objective = cp.Minimize(portfolio_variance)
    constraints = [cp.sum(weights) == 1, weights >= 0, carbon_footprint_constraint]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False, max_iter=100000)
    portfolioWeights = weights.value
    print(f"Step {step}: Status: {problem.status}, Objective value: {problem.value}")

    # Portfolio Performance Evaluation Section
    # Monthly Weights Adjustment
    monthlyReturns = []
    monthlyWeights = [portfolioWeights]
    for singleMonthReturns in evaluationData.values:
        portfolioReturns = monthlyWeights[-1] @ singleMonthReturns
        portfolioWeights = monthlyWeights[-1] * (1 + singleMonthReturns) / (1 + portfolioReturns) # numpy * does element wise multiplication for vectors
        monthlyReturns.append(portfolioReturns)
        monthlyWeights.append(portfolioWeights)
        
    monthlyReturns = pd.Series(monthlyReturns, index=evaluationData.index)

    # Portfolio Value Calculation
    portfolio_value = portfolio_value * (1 + np.array(monthlyReturns)).prod()
    portfolio_values[currentYear] = portfolio_value

    # Portfolio Carbon Footprint Calculation
    meanMonthlyWeights = np.mean(monthlyWeights, axis=0)
    USD_firmValue = meanMonthlyWeights * portfolio_value
    ownership = USD_firmValue / (evaluationCAP.loc[currentYear])
    emissions_xxx = np.array(evaluationEmissions.loc[currentYear])
    carbonFootprint = 1 / portfolio_value * np.sum(ownership * emissions_xxx)
    portfolio_CF[currentYear] = carbonFootprint

    # WACI Calculation
    portfolio_waci[currentYear] = meanMonthlyWeights @ evaluationIntensity.loc[currentYear]

    # Tracking Other Metrics
    portfolio_performance[2006 + step] = portfolio_evaluation(monthlyReturns, evaluationRF)
    portfolio_returns = pd.concat([portfolio_returns, monthlyReturns])
    monthlyWeights = pd.DataFrame(monthlyWeights[1:], index=evaluationData.index, columns=evaluationData.columns)
    monthly_weights.loc[evaluationIndex, monthlyWeights.columns] = monthlyWeights.values

    print('CF:', carbonFootprint)
    print('PV:', portfolio_value)
    print(step, monthlyReturns)

# CSV
save_portfolio_data(portfolio_performance, portfolio_values, portfolio_returns, monthly_weights, portfolio_CF, portfolio_waci, name='BP_50R')

#------------------------------------------------------------#
# 3.1 - Net Zero Portfolio
print('3.1 - Net Zero Portfolio')
# indexIterator = iteration_depth(2006)
target_CF = benchmark_CF.loc['2005-12-31']
reset_portfolio_data()

for step in indexIterator:

    # Loading Data
    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    sampleData = returns[optimizationIndex]
    evaluationData = returns[evaluationIndex]
    evaluationRF = rfRate[evaluationIndex]

    currentYear = pd.to_datetime(f'{step + 2006}-12-31')
    previousYear = pd.to_datetime(f'{step + 2005}-12-31')

    evaluationEmissions = emissions
    evaluationCAP = capData
    evaluationIntensity = intensity

    N = sampleData.shape[1]
    theta = 0.1
    weights = cp.Variable(N)
    covarianceMatrix = expected_covariance(sampleData)
    cap_values = np.array(evaluationCAP.loc[previousYear])
    emissions_values = np.array(evaluationEmissions.loc[previousYear])
    clean_benchmark = benchmark_weights
    benchmark_alpha_Y = np.array(clean_benchmark.loc[previousYear])

    USD_firm_value = cp.multiply(weights, portfolio_value)
    ownership = cp.multiply(USD_firm_value, 1 / cap_values)
    total_emissions = cp.sum(cp.multiply(ownership, emissions_values))
    carbon_footprint_constraint = total_emissions / portfolio_value <= (1 - theta) ** (step+1) * target_CF

    portfolio_variance = cp.quad_form(1e3 * (weights - benchmark_alpha_Y), covarianceMatrix)
    objective = cp.Minimize(portfolio_variance)
    constraints = [cp.sum(weights) == 1, weights >= 0, carbon_footprint_constraint]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    portfolioWeights = weights.value
    print(f"Step {step}: Status: {problem.status}, Objective value: {problem.value}")
    print(evaluationData.shape)

    # Portfolio Performance Evaluation Section
    # Monthly Weights Adjustment
    monthlyReturns = []
    monthlyWeights = [portfolioWeights]
    for singleMonthReturns in evaluationData.values:
        portfolioReturns = monthlyWeights[-1] @ singleMonthReturns
        portfolioWeights = monthlyWeights[-1] * (1 + singleMonthReturns) / (1 + portfolioReturns)
        monthlyReturns.append(portfolioReturns)
        monthlyWeights.append(portfolioWeights)
        
    monthlyReturns = pd.Series(monthlyReturns, index=evaluationData.index)

    # Portfolio Value Calculation
    portfolio_value = portfolio_value * (1 + np.array(monthlyReturns)).prod()
    portfolio_values[currentYear] = portfolio_value

    # Portfolio Carbon Footprint Calculation
    meanMonthlyWeights = np.mean(monthlyWeights, axis=0)
    USD_firmValue = meanMonthlyWeights * portfolio_value
    ownership = USD_firmValue / (evaluationCAP.loc[currentYear])
    emissions_xxx = np.array(evaluationEmissions.loc[currentYear])
    carbonFootprint = 1 / portfolio_value * np.sum(ownership * emissions_xxx)
    portfolio_CF[currentYear] = carbonFootprint

    # WACI Calculation
    portfolio_waci[currentYear] = meanMonthlyWeights @ evaluationIntensity.loc[currentYear]

    # Tracking Other Metrics
    portfolio_performance[2006 + step] = portfolio_evaluation(monthlyReturns, evaluationRF)
    portfolio_returns = pd.concat([portfolio_returns, monthlyReturns])
    monthlyWeights = pd.DataFrame(monthlyWeights[1:], index=evaluationData.index, columns=evaluationData.columns)
    monthly_weights.loc[evaluationIndex, monthlyWeights.columns] = monthlyWeights.values

    print('CF:', carbonFootprint)
    print('PV:', portfolio_value)
    print(step, monthlyReturns)

# CSV
save_portfolio_data(portfolio_performance, portfolio_values, portfolio_returns, monthly_weights, portfolio_CF, portfolio_waci, name='BP_TENZR')