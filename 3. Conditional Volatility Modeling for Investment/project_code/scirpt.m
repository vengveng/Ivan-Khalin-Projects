% Specify the path to the data
info_path = 'info.csv';

% Read the CSV file into a table with specific columns and column names
opts = detectImportOptions(info_path, 'NumHeaderLines', 1);
opts.SelectedVariableNames = [1, 3, 2, 4];
opts.VariableNames = {'date', 'stock', 'bond', 'rate'};
data = readtable(info_path, opts);

% Convert the 'date' column to datetime and set as the index
data.date = datetime(data.date, 'InputFormat', 'yyyy-MM-dd');
data = table2timetable(data);

% Filter the data for dates before 2024
data = data(year(data.date) < 2024, :);

% Adjust the 'rate' for weekly calculation
data.rate = data.rate / (100 * 52);

% Calculate returns and fill missing data
returns = diff(data{:,:}) ./ data{1:end-1,:};
returns = [zeros(1, size(returns, 2)); returns];  % Prepend a row of zeros
returns = fillmissing(returns, 'constant', 0);
returns = array2table(returns, 'VariableNames', {'stock', 'bond', 'rate'});
returns.rate = data.rate;

%-------------------------------------------------------------#
% 1 - Static Allocation

e = ones(2, 1);
returns_1 = [returns.stock, returns.bond]';
R_f = mean(returns.rate);
MU = mean(returns_1, 2);
Sigma = cov(returns_1');
Sigma_inv = inv(Sigma);

static_weights = struct;
for L = [2, 10]
    alpha = 1 / L * Sigma_inv * (MU - R_f);
    alpha_tilde = [alpha; 1 - sum(alpha)];
    static_weights.(['L', num2str(L), '_static']) = alpha_tilde;
end

L2 = static_weights.L2_static;
L10 = static_weights.L10_static;

L2_returns = L2(1) * returns.stock + L2(2) * returns.bond + L2(3) * returns.rate;
L10_returns = L10(1) * returns.stock + L10(2) * returns.bond + L10(3) * returns.rate;

%-------------------------------------------------------------#
% 2 - GARCH
% AR(1) model
% Assuming 'returns' is a timetable with 'stock' and 'bond' as percentage changes
data = returns.Variables;  % Extract matrix of returns data

% Create ARIMA model specifications for an AR(1) model
modelSpec = arima('Constant', NaN, 'ARLags', 1, 'Variance', NaN);

% Estimate the AR(1) model for 'stock'
stock_model = estimate(modelSpec, data(:, 1), 'Display', 'off');
alpha_s = stock_model.Constant;
rho_s = stock_model.AR{1};

% Estimate the AR(1) model for 'bond'
bond_model = estimate(modelSpec, data(:, 2), 'Display', 'off');
alpha_b = bond_model.Constant;
rho_b = bond_model.AR{1};

% Display estimated parameters
fprintf('Stock Model Parameters: alpha = %f, rho = %f\n', alpha_s, rho_s);
fprintf('Bond Model Parameters: alpha = %f, rho = %f\n', alpha_b, rho_b);
% Python AutoReg: 0.0019066003500547285 -0.0768995809552672

% Calculate residuals for 'stock' and 'bond'
epsilon_s = infer(stock_model, data(:, 1));
epsilon_b = infer(bond_model, data(:, 2));

% Drop NaN values (if any) at the beginning due to lag
epsilon_s = epsilon_s(~isnan(epsilon_s));
epsilon_b = epsilon_b(~isnan(epsilon_b));

% Define a GARCH(1,1) model with no mean
model = garch('GARCHLags',1,'ARCHLags',1,'Offset',NaN);

% Fit the GARCH(1,1) model to the stock residuals
[estModelStock, estParamCovStock, logLStock, infoStock] = estimate(model, epsilon_s, 'Display', 'off');
% Fit the GARCH(1,1) model to the bond residuals
[estModelBond, estParamCovBond, logLBond, infoBond] = estimate(model, epsilon_b, 'Display', 'off');

% Extract parameters from the fitted models
stock_params = [estModelStock.Constant; estModelStock.GARCH{1}; estModelStock.ARCH{1}];
bond_params = [estModelBond.Constant; estModelBond.GARCH{1}; estModelBond.ARCH{1}];

% Display the parameters
fprintf('Stock GARCH Parameters: Omega = %f, Beta = %f, Alpha = %f\n', stock_params);
fprintf('Bond GARCH Parameters: Omega = %f, Beta = %f, Alpha = %f\n', bond_params);
% Manual Python LL estimation - IK
%[3.276e-05, 0.23102215, 0.73004919]
%[5.199e-05, 0.17174029, 0.79201544]
% Python Package
% [0.31503262 0.23944924 0.72522888]
% [0.53572521 0.17382131 0.78866245]

%-------------------------------------------------------------#
% 3 - Dynamic Allocation
% Extract residuals from earlier fits and calculate initial covariance and correlation
% epsilon_s = infer(stock_model, data(:, 1));
% epsilon_b = infer(bond_model, data(:, 2));

rho_sb = corr(epsilon_s, epsilon_b);
T = length(epsilon_s);
sigma_s_p1 = 1/T * sum(epsilon_s.^2);
sigma_b_p1 = 1/T * sum(epsilon_b.^2);
sigma_sb_p1 = cov(epsilon_s, epsilon_b);
sigma_sb_p1 = sigma_sb_p1(1, 2);  % Extracting the covariance from the matrix

% Initialize vectors for storing dynamic variances
sigma_s = zeros(T+1, 1);
sigma_b = zeros(T+1, 1);
sigma_sb = zeros(T+1, 1);

% Set initial values
sigma_s(1) = sigma_s_p1;
sigma_b(1) = sigma_b_p1;
sigma_sb(1) = sigma_sb_p1;

% Parameters extracted from GARCH fits
omega_s = stock_params(1);
alpha_s = stock_params(2);
beta_s = stock_params(3);

omega_b = bond_params(1);
alpha_b = bond_params(2);
beta_b = bond_params(3);

for t = 2:T+1
    sigma_s(t) = omega_s + alpha_s * epsilon_s(t-1)^2 + beta_s * sigma_s(t-1);
    sigma_b(t) = omega_b + alpha_b * epsilon_b(t-1)^2 + beta_b * sigma_b(t-1);
    sigma_sb(t) = rho_sb * sqrt(sigma_s(t) * sigma_b(t));
end

returns = [returns.stock, returns.bond];
R_f_t = data(:,3);

% Initialize matrix for dynamic weights
alpha_tilde_dynamic = zeros(T+1, 3);

Lambda = 10;
AR_alpha = [alpha_s; alpha_b];
AR_rho = [rho_s; rho_b];
mu_tp1 = AR_alpha + AR_rho .* returns';  % Element-wise multiplication and transpose
mu_tp1 = mu_tp1';

for t = 1:T
    Sigma = [sigma_s(t), sigma_sb(t); sigma_sb(t), sigma_b(t)];
    Sigma_inv = inv(Sigma);
    ER = mu_tp1(t, :)' - R_f_t(t);
    alpha_tilde_t = 1 / Lambda * Sigma \ (ER);
    alpha_tilde_t = [alpha_tilde_t; 1 - sum(alpha_tilde_t)];
    alpha_tilde_dynamic(t, :) = alpha_tilde_t';
end

% Calculate dynamic portfolio returns
returns = [returns, R_f_t];
dynamic_returns = alpha_tilde_dynamic .* [zeros(1, 3); returns];
dynamic_portfolio_returns = sum(dynamic_returns, 2);
dpr = cumprod(1 + dynamic_portfolio_returns);

figure; % Opens a new figure window
plot(dpr, 'LineWidth', 2); % Plots 'dpr' with a line width of 2 for better visibility
title('Dynamic Portfolio Returns (DPR)'); % Adds a title to the plot
xlabel('Time (Days)'); % Label for the x-axis
ylabel('Cumulative Returns'); % Label for the y-axis
grid on; % Turns on the grid for easier visualization