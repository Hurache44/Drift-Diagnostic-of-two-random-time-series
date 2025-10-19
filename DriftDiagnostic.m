
%%%%
% Drift among time series
%%%

% Drift Diagnostics
% my version
% Result % p value < 0.05 significant drift 
% Drift results alone still unvalid, both time series drift detected!

%%%
% Understanding the ACF and PACF: values with higher altitudes tend to
% have a positive effect on stock evaluations while values around zero for
% a constant time seem to suggest downward potential for further periods

% ACF: correlation between r(t) and r(t-k) for each lag k; values near
% zero, returns are roughly uncorrelated (white noise), significant spikes
% indicate persistence or mean reversion

% PACF: measures direct correlation between r(t) and r(t-k) removing
% effects of intermediate lags; identifying true order of autoregressive
% (AR) behaviour, AR(1) if PACF cuts off sharply after lag 1

%%%

clear;
close all;
clc;

% Time series 1 and 2
% 1
n_steps = 3600;
dt = 1;
step_size = 0.1;
x = zeros(1,n_steps);
%%%% Starting point
drift_x = [0.001];
for i = 2:n_steps
    x(1) = 12;
    x(i) = x(i-1) + drift_x + step_size * randn();
end
% disp('The simulated price of the x asset is: ')
% disp(x(end));

% 2
n_steps = 3600;
step_size = 0.1;
x2 = zeros(1,n_steps);
%%%% Starting point
drift_x2 = [0.001];
for i = 2:n_steps
    x2(1) = 15;
    x2(i) = x2(i-1) + drift_x2 + step_size * randn();
end
% disp('The simulated price of the asset x2 is: ')
% disp(x2(end));

% 3
n_steps = 3600;
step_size = 0.1;
x3 = zeros(1,n_steps);
%%%% Starting point
drift_x3 = [0.008];
for i = 2:n_steps
    x3(1) = 0.196;
    x3(i) = x3(i-1) + drift_x3 + step_size * randn();
end
disp('The simulated asset prices: ')
fprintf('x: %.4f | x2: %.4f | x3: %.4f\n', x(end), x2(end), x3(end));
%% Data from brownian_motion_Data.m
% PF0/PF1/PF2: Asset 1/2/3

% reading data
data = readtable('/Users/robertlange/Desktop/Investment/Matlab/Returns.xlsx');
date = table2array(data(:,1));
PF_0 = table2array(data(:,"Oatly"));
PF_1 = table2array(data(:,"Prada"));
PF_2 = table2array(data(:,"WaltDisney"));
PF_3 = table2array(data(:,"Lufthansa"));
PF_4 = table2array(data(:,"NovoNordisk"));
PF_5 = table2array(data(:,"UnitedHealth"));
PF_6 = table2array(data(:,"Papoutsanis"));
PF_7 = table2array(data(:,"Tencent"));
PF_8 = table2array(data(:,"PorscheHoldingSE"));
PF_9 = table2array(data(:,"MorganAdv_"));
PF_10 = table2array(data(:,"Burberry"));
PF_11 = table2array(data(:,"HugoBoss"));
PF_12 = table2array(data(:,"Ubisoft"));
PF_13 = table2array(data(:,"Citigroup"));
PF_14 = table2array(data(:,"Nike"));
PF_15 = table2array(data(:,"WarnerBros_"));
PF_16 = table2array(data(:,"Lululemon"));

X = [PF_0 PF_1 PF_2 PF_3 PF_4 PF_5 PF_6 PF_7 PF_8 PF_9 PF_10 PF_11 PF_12 PF_13 PF_14 PF_15];

PF = [PF_0,PF_5,PF_6,PF_7];

R1 = diff(log(PF_0));
R2 = diff(log(PF_5));
R3 = diff(log(PF_6));
R4 = diff(log(PF_7));

% not hardcoding
% selectedCols = [1 6 7 8];
% PF = X(:,selectedCols);
nSeries = size(PF,2);
% returns = cell(1,nSeries);
% for i = 1:nSeries
%  returns{i} = diff(log(PF(:,i)));
%  end
% names = data.Properties.VariableNames(selectedCols);

% Drift example
% base = returns{1};
% results = cell(1,nSeries-1);
% for j = 2:nSeries
%     results{j-1} = detectdrift(base,returns{j});
%     fprintf('-- Drift Detection R1 vs %s -- \n', names{j});
%     disp(results{j-1});
%     disp(results{j-1}.ConfidenceIntervals);
% end

% Exmaple Plotting
% figure('Name','Permutation Results','NumberTitle','off');
% tiledlayout(2,ceil((nSeries-1)/2));
% for j = 2:nSeries
%     nexttile;
%     plotPermutationResults(results{j-1});
%     title(['R1 vs ', names{j}]);
% end

%%

% Returns of time series
% R1 = diff(x)./x(1:end-1); 
% R2 = diff(x2)./x2(1:end-1);
% R3 = diff(x3)./x3(1:end-1); 

% Values for baseline and target 
R1 = R1(:);
R2 = R2(:);
R3 = R3(:);
R4 = R4(:);
%%
disp('Mean returns: ');
fprintf('R1 = %.6f | R2 = %.6f | R3 = %.6f | R4 = %.6f\n', mean(R1),mean(R2),mean(R3),mean(R4));
%%

%%%% Stationarity tests 
% ADF KPSS

fprintf('\n--- Stationarity Tests (ADF/KPSS) ---\n');
[h_adf_r1,p_adf_r1] = adftest(R1);
[h_kpss_r1,p_kss_r1] = kpsstest(R1);
fprintf('R1: ADF h=%d p=5f | KPSS h=%d p=%.4f\n',h_adf_r1,p_adf_r1,h_kpss_r1,p_kss_r1);

[h_adf_r2,p_adf_r2] = adftest(R2);
[h_kpss_r2,p_kss_r2] = kpsstest(R2);
fprintf('R2: ADF h=%d p=5f | KPSS h=%d p=%.4f\n',h_adf_r2,p_adf_r2,h_kpss_r2,p_kss_r2);

[h_adf_r3,p_adf_r3] = adftest(R3);
[h_kpss_r3,p_kss_r3] = kpsstest(R3);
fprintf('R3: ADF h=%d p=5f | KPSS h=%d p=%.4f\n',h_adf_r3,p_adf_r3,h_kpss_r3,p_kss_r3);

% Values are not displayed for KPSS
[h_adf_r4,p_adf_r4] = adftest(R4);
[h_kpss_r4,p_kss_r4] = kpsstest(R4);
fprintf('R4: ADF h=%d p=5f | KPSS h=%d p=%.4f\n',h_adf_r4,p_adf_r4,h_kpss_r4,p_kss_r4);

%%%%%% White Noise Diagnostics
%
fprintf('\n--- White Noise Tests ---\n');
seriesNames = {'R1','R2','R3','R4'};
returns = {R1,R2,R3,R4};

for i = 1:numel(returns)
    r = returns{i}(:);
    [h_lbq,p_lbq] = lbqtest(r,'Lags',20); % Ljung Box test up to lag 20
    r = r(:);
    acf_vals = autocorr(r,'NumLags',1);
    rho1 = acf_vals(2);
    fprintf('%s: Ljung Box h = %d p = %4f | lag1 autocorr = %.4f\n', ...
        char(seriesNames{i}),h_lbq,p_lbq,rho1);
end
fprintf('\n');

% Visualization
figure;
tiledlayout(2,2);
for i = 1:4
    nexttile;
    autocorr(returns{i}(:));
    title(['ACF ',seriesNames{i}]);
end

figure;
tiledlayout(2,2);
for i = 1:4
    nexttile;
    parcorr(returns{i}(:));
    title(['PACF ',seriesNames{i}]);
end
%%
%%%%% Drift Detection
%
% baseline = [R1];
% target = [R2/R3];
result = detectdrift([R1 R2],[R1 R3]);
disp(result);

% Drift Diagnostics
% result = detectdrift(baseline, target);
result1 = detectdrift(R1, R2);
result2 = detectdrift(R1, R3);
result3 = detectdrift(R1, R4);
disp(result1);
disp('Confidence Intervals (R1 vs R2): ');
disp(result1.ConfidenceIntervals);
disp(result2);
disp('Confidence Intervals (R1 vs R3): ');
disp(result2.ConfidenceIntervals);
disp(result3);
disp('Confidence Intervals (R1 vs R4): ');
disp(result3.ConfidenceIntervals);
%%
%%%%% Visualization

figure;
tiledlayout(3,3);
ax1 = nexttile;
plotPermutationResults(result1,ax1);
title('Permutation Results Series 1, R1 vs. R2: ');
ax2 = nexttile;
plotPermutationResults(result2,ax2);
title('Permutation Results Series 2, R1 vs. R3: ');
ax3 = nexttile;
plotPermutationResults(result3,ax3);
title('Permutation Results Series 3, R1 vs. R4: ');

ax4 = nexttile;
plotDriftStatus(result1);
hold on;
stable_value = mean(R2(:));
yline(stable_value,'--r','LineWidth',1.5,'DisplayName','Stable Allocation');
hold on;
plot(1, stable_value,'ob','MarkerSize',6,'LineWidth',1.5,'DisplayName','Start Point');
legend('Stable Allocation','Drift Status','Drift Treshold','Warning Treshold');
title('Drift Status with stable Allocation');

ax5 = nexttile;
plotDriftStatus(result2);
hold on;
stable_value = mean(R3(:));
yline(stable_value,'--r','LineWidth',1.5,'DisplayName','Stable Allocation');
hold on;
plot(1, stable_value,'ob','MarkerSize',6,'LineWidth',1.5,'DisplayName','Start Point');
legend('Stable Allocation','Drift Status','Drift Treshold','Warning Treshold');
title('Drift Status with stable Allocation');

ax6 = nexttile;
plotDriftStatus(result3);
hold on;
stable_value = mean(R4(:));
yline(stable_value,'--r','LineWidth',1.5,'DisplayName','Stable Allocation');
hold on;
plot(1, stable_value,'ob','MarkerSize',6,'LineWidth',1.5,'DisplayName','Start Point');
legend('Stable Allocation','Drift Status','Drift Treshold','Warning Treshold');
title('Drift Status with stable Allocation');

figure;
plotEmpiricalCDF(result1,'Variable', 'x1');
hold on;
plotEmpiricalCDF(result2,'Variable', 'x1');

%%%%% Drift Significance Test

fprintf('\n--- Drift Mean Test (autocorr adjusted) ---\n');
[t_r1,p_r1,se_r1] = mean_test_autocorr_adjusted(R1);
[t_r2,p_r2,se_r2] = mean_test_autocorr_adjusted(R2);
[t_r3,p_r3,se_r3] = mean_test_autocorr_adjusted(R3);
fprintf('R1: t = %.3f p = %.4f mean = %.6g se = %.6g\n',t_r1,p_r1,mean(R1),se_r1);
fprintf('R2: t = %.3f p = %.4f mean = %.6g se = %.6g\n',t_r2,p_r2,mean(R2),se_r2);
fprintf('R3: t = %.3f p = %.4f mean = %.6g se = %.6g\n',t_r3,p_r3,mean(R3),se_r3);
%%
%%% Diagnostic Plots
figure('Name','Comprehensive Diagnostics','NumberTitle','off');
tiledlayout(4,4,'TileSpacing','compact','Padding','compact');
names = {'R1','R2','R3','R4'};

for i = 1:4
    r = returns{i};
    % Price path
    nexttile((i-1)*4+1);
    plot(PF(1:numel(r)+1,i),'k','LineWidth',1.2);
    title([names{i} ' Price Path']);
    xlabel('Time');
    ylabel('Price');

    nexttile((i-1)*4+2);
    plot(r,'b');
    title([names{i},'Returns']);
    xlabel('t');
    ylabel('log return');

    nexttile((i-1)*4+3);
    histogram(r,40,'Normalization','pdf','FaceColor',[0.2 0.6 0.8]);
    hold on;
    mu = mean(r);
    sigma = std(r);
    xgrid = linspace(min(r),max(r),200);
    plot(xgrid,normpdf(xgrid,mu,sigma),'r--','LineWidth',1.2);
    title(['Histogram & Normal Fit( ' names{i} ')']);

    nexttile((i-1)*4+4);
    qqplot(r);
    title(['QQ Plot (' names{i} ')']);
end
%%
figure('Name','Autocorrelation (ACF/PACF)','NumberTitle','off');
tiledlayout(2,4);
for i = 1:4
    nexttile(i);
    autocorr(returns{i});
    title(['ACF ',names{i}]);
end
for i = 1:4
    nexttile(i+4);
    parcorr(returns{i});
    title(['PACF ', names{i}]);
end

[t_r1,p_r1,se_r1] = mean_test_autocorr_adjusted(R1);

function [tstat,pval,se_adj] = mean_test_autocorr_adjusted(r)
r = r(:);
N = length(r);
rbar = mean(r);
s = std(r,1);
acf_vals = autocorr(r,'NumLags',1);
rho1 = acf_vals(2);
N_eff = N*(1-rho1)/(1+rho1);
se_adj = s/sqrt(max(N_eff,1));
tstat = rbar/se_adj;
pval = 2*(1-tcdf(abs(tstat),max(1,floor(N_eff)-1)));
end


%% -- Interpretation of ACF & PACF Results --
disp('');
disp('--- Interpretation of Autocorrelation Structure ---');

for i = 1:numel(returns)
    r = returns{i}(:);
    [acf_vals,~,acf_bounds] = autocorr(r,'NumLags',20);
    [pacf_vals,~,pacf_bounds] = parcorr(r,'NumLags',20);

    sig_acf = abs(acf_vals(2:end)) > acf_bounds(2);
    sig_pacf = abs(pacf_vals(2:end)) > pacf_bounds(2);

    fprintf('\nSeries %s:\n',names{i});

    if ~any(sig_acf) && ~any(sig_pacf)
        fprintf(' ACF and PACF both flat -> series behaves like white noise (no predictable structure).\n');
    elseif acf_vals(2) > 0 && sig_acf(1)
        fprintf(' ACF positive at lag 1 -> persistence or short-term momentum effect.\n');
    elseif acf_vals(2) < 0 && sig_acf(1)
        fprintf(' ACF negative at lag 1 -> mean-reverting behaviour (returns tend to reverse direction).\n');
    end

    if sig_pacf(1)
        fprintf(' PACF significant at lag 1 -> possible AR(1)-type dependency.\n');
    elseif sig_pacf(2)
        fprintf(' PACF significant at lag 2 -> possible AR(2) structure.\n');
    elseif ~any(sig_pacf)
        fprintf(' PACF flat -> no direct autoregressive structure detected.\n');
    end

    % Combined reasoning
    if all(abs(acf_vals(2:5)) < 0.1) && all(abs(pacf_vals(2:5)) < 0.1)
        fprintf(' Short-term correlations negligible -> resembles efficient market / random walk.\n');
    elseif mean(acf_vals(2:5)) > 0.15
        fprintf(' Moderate persistence across first few lags -> potential trend following behaviour.\n');
    elseif mean(acf_vals(2:5)) < -0.15
        fprintf(' Negative autocorrelation in first few lags -> strong mean reversion.\n');
    end
end

disp(' ');
disp('-- End of Interpretation --');

%% Visualization Mean ACF and PACF Diagnostics
maxLag = 20;
meanACF = zeros(nSeries,1);
meanPACF = zeros(nSeries,1);

for i = 1:nSeries
    acf_vals = autocorr(returns{i},'NumLags',maxLag);
    pacf_vals = parcorr(returns{i},'NumLags',maxLag);

    % Skip lag one
    meanACF(i) = mean(acf_vals(2:end));
    meanPACF(i) = mean(pacf_vals(2:end));
end

figure('Name','Mean ACF and PACF per Series','NumberTitle','off');
bar([meanACF meanPACF]);
xticklabels(names);
legend({'Mean ACF','Mean PACF'},'Location','northwest');
ylabel('Average Correlation');
title('Mean ACF and PACF Strength per Time Series');
grid on;

% Interpretation
txt = {
    'Interpretation Guide;'
    '‹ High mean ACF -> persistence or trend-following'
    '‹ High mean PACF but low ACF -> direct AR(1) structure'
    '‹ Both near zero -> white-noise-like (no predictable pattern'
    '‹ Alternating signs -> mean reversion /  oscillatory behaviour'};

annotation('textbox',[0.55,0.15,0.4,0.3], ...
    'String',txt, ...
    'FitBoxToText','on', ...
    'EdgeColor','none', ...
    'FontSize',10, ...
    'BackgroundColor',[0.95 0.95 0.95], ...
    'Interpreter','none');
