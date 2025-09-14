
%%%%
% Drift among time series
%%%

% Drift Diagnostics
% my version
% Result % p value < 0.05 significant drift 

% Time series 1 and 2
% 1
n_steps = 3600;
step_size = 0.1;
x = zeros(1,n_steps);
%%%% Starting point
drift_x = [0.001];
for i = 2:n_steps
    x(1) = 12;
    x(i) = x(i-1) + drift_x + step_size * randn();
end
disp('The simulated price of the x asset is: ')
disp(x(end));

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
disp('The simulated price of the asset x2 is: ')
disp(x2(end));

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
disp('The simulated price of the asset x3 is: ')
disp(x3(end));
%%

% Returns of time series
% R1 = diff(x)./x(1:end-1); 
% R2 = diff(x2)./x2(1:end-1);
% R3 = diff(x3)./x3(1:end-1); 

% Values for baseline and target 
R1 = x(:);
R2 = x2(:);
R3 = x3(:);

% baseline = [R1];
% target = [R2/R3];
result = detectdrift([R1 R2],[R1 R3]);
disp(result);

% Drift Diagnostics
% result = detectdrift(baseline, target);
result1 = detectdrift(R1, R2);
result2 = detectdrift(R1, R3);
disp(result1);
disp('Confidence Intervals: ');
disp(result1.ConfidenceIntervals);
disp(result2);
disp('Confidence Intervals: ');
disp(result2.ConfidenceIntervals);

figure;
tiledlayout(2,2);
ax1 = nexttile;
plotPermutationResults(result1,ax1);
title('Permutation Results Series 1, R1 vs. R2: ');
ax2 = nexttile;
plotPermutationResults(result2,ax2);
title('Permutation Results Series 2, R1 vs. R3: ');
ax3 = nexttile;
plotDriftStatus(result1);
hold on;
stable_value = mean(R2(:));
yline(stable_value,'--r','LineWidth',1.5,'DisplayName','Stable Allocation');
hold on;
plot(1, stable_value,'ob','MarkerSize',6,'LineWidth',1.5,'DisplayName','Start Point');
legend('Stable Allocation','Drift Status','Drift Treshold','Warning Treshold');
title('Drift Status with stable Allocation');

ax4 = nexttile;
plotDriftStatus(result2);
hold on;
stable_value = mean(R2(:));
yline(stable_value,'--r','LineWidth',1.5,'DisplayName','Stable Allocation');
hold on;
plot(1, stable_value,'ob','MarkerSize',6,'LineWidth',1.5,'DisplayName','Start Point');
legend('Stable Allocation','Drift Status','Drift Treshold','Warning Treshold');
title('Drift Status with stable Allocation');

figure;
plotEmpiricalCDF(result1,Variable="x1");
hold on;
plotEmpiricalCDF(result2,Variable="x1");
