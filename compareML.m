
%% 1. Load Data
X_data = load('inputData.mat');
X_all = X_data.inputMat;

real_data = load('outputData1.mat');
Y_x_all = real_data.outputMat1;

imag_data = load('outputData2.mat');
Y_y_all = imag_data.outputMat2;
%% --- Add History (Delta / Rate of Change) ---
[numSamples, numFeatures] = size(X_all);

% Pre-allocation (one previous time step is enough for difference calculation)
X_enhanced = zeros(numSamples - 1, numFeatures * 2);
Y_x_enhanced = Y_x_all(2 : end, :);
Y_y_enhanced = Y_y_all(2 : end, :);

for i = 2 : numSamples
    current_meas = X_all(i, :);
    prev_meas = X_all(i-1, :);
    
    delta_meas = current_meas - prev_meas; % Calculate rate of change (slope)
    
    % Combine current measurement with rate of change
    X_enhanced(i-1, :) = [current_meas, delta_meas];
end

X_all = X_enhanced;
Y_x_all = Y_x_enhanced;
Y_y_all = Y_y_enhanced;
numSamples = size(X_all, 1);
%% ------------------------------------
%% 2. Shuffle Data (Very Important for Generalization)
numSamples = size(X_all, 1);
rng(42,'twister'); % Set seed for reproducibility (optional)
% Split into Train and Test based on time sequence (without shuffling)
trainRatio = 0.9;
numTrain = floor(trainRatio * numSamples);

% First, separate the training data (first 90% of the path)
X_train = X_all(1:numTrain, :);
Y_x_train = Y_x_all(1:numTrain, :);
Y_y_train = Y_y_all(1:numTrain, :);

% Then separate the test data (last 10% of the path)
X_test = X_all(numTrain+1:end, :);
Y_x_test = Y_x_all(numTrain+1:end, :);
Y_y_test = Y_y_all(numTrain+1:end, :);

% Now only shuffle the training data
idx_train = randperm(numTrain);
X_train = X_train(idx_train, :);
Y_x_train = Y_x_train(idx_train, :);
Y_y_train = Y_y_train(idx_train, :);


%% Arrays to store MSE for final comparison
mse_x_results = zeros(1, 4);
mse_y_results = zeros(1, 4);
model_names = {'SVM', 'ANN', 'Decision Tree', 'Ensemble'};

%% ========================================================
%% 1. Support Vector Machine (SVM)
disp('--- Training SVM ---');
% Using Gaussian kernel and Standardization for better non-linear fitting
model_x_svm = fitrsvm(X_train, Y_x_train, 'Standardize', true, 'KernelFunction', 'gaussian');
model_y_svm = fitrsvm(X_train, Y_y_train, 'Standardize', true, 'KernelFunction', 'gaussian');

% Cross Validation
loss_x_svm = kfoldLoss(crossval(model_x_svm, 'KFold', 10));
loss_y_svm = kfoldLoss(crossval(model_y_svm, 'KFold', 10));

% Prediction & MSE
pred_x_svm = predict(model_x_svm, X_test);
pred_y_svm = predict(model_y_svm, X_test);
mse_x_results(1) = immse(Y_x_test, pred_x_svm);
mse_y_results(1) = immse(Y_y_test, pred_y_svm);

disp(['SVM X-coordinate CV Loss: ', num2str(loss_x_svm), ' | Test MSE: ', num2str(mse_x_results(1))]);
disp(['SVM Y-coordinate CV Loss: ', num2str(loss_y_svm), ' | Test MSE: ', num2str(mse_y_results(1))]);

%% ========================================================
%% 2. Artificial Neural Network (ANN)
disp('--- Training ANN ---');
% Standardization is crucial for neural networks
model_x_ann = fitrnet(X_train, Y_x_train, 'Standardize', true);
model_y_ann = fitrnet(X_train, Y_y_train, 'Standardize', true);

% Cross Validation
loss_x_ann = kfoldLoss(crossval(model_x_ann, 'KFold', 10));
loss_y_ann = kfoldLoss(crossval(model_y_ann, 'KFold', 10));

% Prediction & MSE
pred_x_ann = predict(model_x_ann, X_test);
pred_y_ann = predict(model_y_ann, X_test);
mse_x_results(2) = immse(Y_x_test, pred_x_ann);
mse_y_results(2) = immse(Y_y_test, pred_y_ann);

disp(['ANN X-coordinate CV Loss: ', num2str(loss_x_ann), ' | Test MSE: ', num2str(mse_x_results(2))]);
disp(['ANN Y-coordinate CV Loss: ', num2str(loss_y_ann), ' | Test MSE: ', num2str(mse_y_results(2))]);

%% ========================================================
%% 3. Decision Tree (DT)
disp('--- Training Decision Tree ---');
model_x_dt = fitrtree(X_train, Y_x_train);
model_y_dt = fitrtree(X_train, Y_y_train);

% Cross Validation
loss_x_dt = kfoldLoss(crossval(model_x_dt, 'KFold', 10));
loss_y_dt = kfoldLoss(crossval(model_y_dt, 'KFold', 10));

% Prediction & MSE
pred_x_dt = predict(model_x_dt, X_test);
pred_y_dt = predict(model_y_dt, X_test);
mse_x_results(3) = immse(Y_x_test, pred_x_dt);
mse_y_results(3) = immse(Y_y_test, pred_y_dt);

disp(['DT X-coordinate CV Loss: ', num2str(loss_x_dt), ' | Test MSE: ', num2str(mse_x_results(3))]);
disp(['DT Y-coordinate CV Loss: ', num2str(loss_y_dt), ' | Test MSE: ', num2str(mse_y_results(3))]);

%% ========================================================
%% 4. Ensemble Learning (EL)
disp('--- Training Ensemble Learning ---');
model_x_el = fitrensemble(X_train, Y_x_train);
model_y_el = fitrensemble(X_train, Y_y_train);

% Cross Validation
loss_x_el = kfoldLoss(crossval(model_x_el, 'KFold', 10));
loss_y_el = kfoldLoss(crossval(model_y_el, 'KFold', 10));

% Prediction & MSE
pred_x_el = predict(model_x_el, X_test);
pred_y_el = predict(model_y_el, X_test);
mse_x_results(4) = immse(Y_x_test, pred_x_el);
mse_y_results(4) = immse(Y_y_test, pred_y_el);

disp(['EL X-coordinate CV Loss: ', num2str(loss_x_el), ' | Test MSE: ', num2str(mse_x_results(4))]);
disp(['EL Y-coordinate CV Loss: ', num2str(loss_y_el), ' | Test MSE: ', num2str(mse_y_results(4))]);



%% ========================================================
%% 5. Plotting Comparisons (Updated for 4 Models)
model_names = {'SVM', 'ANN', 'DT', 'Ensemble'};

% Construction of arrays of CV Loss 
cv_loss_x = [loss_x_svm, loss_x_ann, loss_x_dt, loss_x_el];
cv_loss_y = [loss_y_svm, loss_y_ann, loss_y_dt, loss_y_el];

% ---------------------------------------------------------
% Plot X Coordinate Predictions (4 Models)
figure('Name','X-Coordinate Predictions','NumberTitle','off');
subplot(2,2,1); plot(Y_x_test,'-bs'); hold on; plot(pred_x_svm,'-or'); hold off; title(['SVM X (MSE: ', num2str(mse_x_results(1), '%.1f'), ')']); legend('Desired','Predicted');
subplot(2,2,2); plot(Y_x_test,'-bs'); hold on; plot(pred_x_ann,'-or'); hold off; title(['ANN X (MSE: ', num2str(mse_x_results(2), '%.1f'), ')']);
subplot(2,2,3); plot(Y_x_test,'-bs'); hold on; plot(pred_x_dt,'-or'); hold off; title(['DT X (MSE: ', num2str(mse_x_results(3), '%.1f'), ')']);
subplot(2,2,4); plot(Y_x_test,'-bs'); hold on; plot(pred_x_el,'-or'); hold off; title(['Ensemble X (MSE: ', num2str(mse_x_results(4), '%.1f'), ')']);
print('-dpng','-r300','X-Coordinate Predictions');
% ---------------------------------------------------------
% Plot Y Coordinate Predictions (4 Models)
figure('Name','Y-Coordinate Predictions','NumberTitle','off');
subplot(2,2,1); plot(Y_y_test,'-bs'); hold on; plot(pred_y_svm,'-or'); hold off; title(['SVM Y (MSE: ', num2str(mse_y_results(1), '%.1f'), ')']); legend('Desired','Predicted');
subplot(2,2,2); plot(Y_y_test,'-bs'); hold on; plot(pred_y_ann,'-or'); hold off; title(['ANN Y (MSE: ', num2str(mse_y_results(2), '%.1f'), ')']);
subplot(2,2,3); plot(Y_y_test,'-bs'); hold on; plot(pred_y_dt,'-or'); hold off; title(['DT Y (MSE: ', num2str(mse_y_results(3), '%.1f'), ')']);
subplot(2,2,4); plot(Y_y_test,'-bs'); hold on; plot(pred_y_el,'-or'); hold off; title(['Ensemble Y (MSE: ', num2str(mse_y_results(4), '%.1f'), ')']);
print('-dpng','-r300','Y-Coordinate Predictions');
% ---------------------------------------------------------
% --- Style Definitions for Professional Plots ---
fontName = 'Times New Roman';
axisFontSize = 12;
titleFontSize = 14;
labelFontSize = 10;
color1 = [0.12, 0.47, 0.71]; % Professional Blue
color2 = [0.85, 0.33, 0.10]; % Burnt Orange
gridAlpha = 0.3; % Grid line transparency
barWidth = 0.8;

% Data preparation
data_mse = [mse_x_results(1:4); mse_y_results(1:4)]';
data_cv = [cv_loss_x; cv_loss_y]';

%% ---------------------------------------------------------
% 1. PUBLICATION-READY: Test MSE Comparison (Line & Bar)
figure('Name','Test MSE Comparison (Professional)','NumberTitle','off', 'Color', 'w');
set(gcf, 'Position', [100, 100, 800, 600]); 

% --- Top Subplot: Line Chart ---
subplot(2, 1, 1);
plot(1:numel(model_names), data_mse(:, 1), '-o', 'Color', color1, 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', color1);
hold on;
plot(1:numel(model_names), data_mse(:, 2), '-s', 'Color', color2, 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', color2);
ax1 = gca;
ax1.FontName = fontName; ax1.FontSize = axisFontSize;
ax1.XTick = 1:numel(model_names); ax1.XTickLabel = model_names; ax1.FontWeight = 'bold';
ax1.YGrid = 'on'; ax1.GridLineStyle = '--'; ax1.GridAlpha = gridAlpha; ax1.Box = 'on'; ax1.LineWidth = 1.2;
ylabel('Test MSE', 'FontName', fontName, 'FontSize', axisFontSize, 'FontWeight', 'bold');
title('Performance Comparison of Algorithms on Test Data', 'FontName', fontName, 'FontSize', titleFontSize, 'FontWeight', 'bold');
legend('X-Coordinate', 'Y-Coordinate', 'Location', 'best', 'FontName', fontName, 'FontSize', labelFontSize);

% --- Bottom Subplot: Bar Chart ---
subplot(2, 1, 2);
b1 = bar(data_mse, barWidth);
b1(1).FaceColor = color1; b1(2).FaceColor = color2;
b1(1).EdgeColor = 'k';    b1(2).EdgeColor = 'k';
ax2 = gca;
ax2.FontName = fontName; ax2.FontSize = axisFontSize;
ax2.XTick = 1:numel(model_names); ax2.XTickLabel = model_names; ax2.FontWeight = 'bold';
ax2.YGrid = 'on'; ax2.GridLineStyle = '--'; ax2.GridAlpha = gridAlpha; ax2.Box = 'on'; ax2.LineWidth = 1.2;
ylabel('Test MSE', 'FontName', fontName, 'FontSize', axisFontSize, 'FontWeight', 'bold');
legend('X-Coordinate', 'Y-Coordinate', 'Location', 'best', 'FontName', fontName, 'FontSize', labelFontSize);

% Add data labels
for i = 1:numel(b1)
    xtips = b1(i).XEndPoints; ytips = b1(i).YEndPoints;
    yoffset = 0.05 * max(data_mse(:)); 
    labels = string(round(ytips));
    text(xtips, ytips + yoffset, labels, 'HorizontalAlignment','center',...
        'VerticalAlignment','bottom', 'FontName', fontName, 'FontSize', labelFontSize-1, 'FontWeight','bold');
end
print('-dpng','-r300','Test MSE Comparison (Line & Bar)');
%% ---------------------------------------------------------
% 2. PUBLICATION-READY: CV Loss Comparison (Line & Bar)
figure('Name','CV Loss Comparison (Professional)','NumberTitle','off', 'Color', 'w');
set(gcf, 'Position', [150, 150, 800, 600]); 

% --- Top Subplot: Line Chart ---
subplot(2, 1, 1);
plot(1:numel(model_names), data_cv(:, 1), '-o', 'Color', color1, 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', color1);
hold on;
plot(1:numel(model_names), data_cv(:, 2), '-s', 'Color', color2, 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', color2);
ax3 = gca;
ax3.FontName = fontName; ax3.FontSize = axisFontSize;
ax3.XTick = 1:numel(model_names); ax3.XTickLabel = model_names; ax3.FontWeight = 'bold';
ax3.YGrid = 'on'; ax3.GridLineStyle = '--'; ax3.GridAlpha = gridAlpha; ax3.Box = 'on'; ax3.LineWidth = 1.2;
ylabel('CV Loss', 'FontName', fontName, 'FontSize', axisFontSize, 'FontWeight', 'bold');
title('Generalization Error Comparison (10-Fold CV)', 'FontName', fontName, 'FontSize', titleFontSize, 'FontWeight', 'bold');
legend('X-Coordinate', 'Y-Coordinate', 'Location', 'best', 'FontName', fontName, 'FontSize', labelFontSize);

% --- Bottom Subplot: Bar Chart ---
subplot(2, 1, 2);
b2 = bar(data_cv, barWidth);
b2(1).FaceColor = color1; b2(2).FaceColor = color2;
b2(1).EdgeColor = 'k';    b2(2).EdgeColor = 'k';
ax4 = gca;
ax4.FontName = fontName; ax4.FontSize = axisFontSize;
ax4.XTick = 1:numel(model_names); ax4.XTickLabel = model_names; ax4.FontWeight = 'bold';
ax4.YGrid = 'on'; ax4.GridLineStyle = '--'; ax4.GridAlpha = gridAlpha; ax4.Box = 'on'; ax4.LineWidth = 1.2;
ylabel('CV Loss', 'FontName', fontName, 'FontSize', axisFontSize, 'FontWeight', 'bold');
legend('X-Coordinate', 'Y-Coordinate', 'Location', 'best', 'FontName', fontName, 'FontSize', labelFontSize);

% Add data labels
for i = 1:numel(b2)
    xtips = b2(i).XEndPoints; ytips = b2(i).YEndPoints;
    yoffset = 0.05 * max(data_cv(:)); 
    labels = string(round(ytips));
    text(xtips, ytips + yoffset, labels, 'HorizontalAlignment','center',...
        'VerticalAlignment','bottom', 'FontName', fontName, 'FontSize', labelFontSize-1, 'FontWeight','bold');
end
print('-dpng','-r300','CV Loss Comparison (Line & Bar)');
%% ========================================================
%% 6. Calculate Percentage Difference of ANN Compared to Others
disp(' ');
disp('=====================================================================================');
disp('   Percentage Difference (%) in Error of Artificial Neural Network (ANN) Compared to Other Models');
disp('   (Positive values mean the other model has HIGHER error than ANN / ANN performs better)');
disp('=====================================================================================');

% Index of Artificial Neural Network (ANN) in the arrays is 2
ann_idx = 2; 
% Indices of models to compare against: SVM (1), Decision Tree (3), Ensemble (4)
compare_indices = [1, 3, 4];
Compared_With = {'SVM'; 'Decision Tree'; 'Ensemble'};

% Preallocate arrays for percentages
MSE_X_Diff_Percent = zeros(3, 1);
MSE_Y_Diff_Percent = zeros(3, 1);
CV_Loss_X_Diff_Percent = zeros(3, 1);
CV_Loss_Y_Diff_Percent = zeros(3, 1);

% Calculate percentages
for i = 1:3
    idx = compare_indices(i);
    
    % MSE Difference (%)
    % Formula: ((Error_Other - Error_ANN) / Error_Other) * 100
    MSE_X_Diff_Percent(i) = ((mse_x_results(idx) - mse_x_results(ann_idx)) / mse_x_results(idx)) * 100;
    MSE_Y_Diff_Percent(i) = ((mse_y_results(idx) - mse_y_results(ann_idx)) / mse_y_results(idx)) * 100;
    
    % CV Loss Difference (%)
    CV_Loss_X_Diff_Percent(i) = ((cv_loss_x(idx) - cv_loss_x(ann_idx)) / cv_loss_x(idx)) * 100;
    CV_Loss_Y_Diff_Percent(i) = ((cv_loss_y(idx) - cv_loss_y(ann_idx)) / cv_loss_y(idx)) * 100;
end

% Round values for a cleaner display
MSE_X_Diff_Percent = round(MSE_X_Diff_Percent, 2);
MSE_Y_Diff_Percent = round(MSE_Y_Diff_Percent, 2);
CV_Loss_X_Diff_Percent = round(CV_Loss_X_Diff_Percent, 2);
CV_Loss_Y_Diff_Percent = round(CV_Loss_Y_Diff_Percent, 2);

% Create and display the table
ANN_Comparison_Table = table(Compared_With, MSE_X_Diff_Percent, MSE_Y_Diff_Percent, ...
    CV_Loss_X_Diff_Percent, CV_Loss_Y_Diff_Percent);

disp(ANN_Comparison_Table);
disp('=====================================================================================');
%% 7. Calculate Percentage Difference of SVM Compared to Others
disp(' ');
disp('=====================================================================================');
disp('   Percentage Difference (%) in Error of Support Vector Machine (SVM) Compared to Other Models');
disp('   (Positive values mean the other model has HIGHER error than SVM / SVM performs better)');
disp('=====================================================================================');

% Index of Support Vector Machine (SVM) in the arrays is 1
svm_idx = 1; 
% Indices of models to compare against: ANN (2), Decision Tree (3), Ensemble (4)
compare_indices = [2, 3, 4];
Compared_With = {'ANN'; 'Decision Tree'; 'Ensemble'};

% Preallocate arrays for percentages
MSE_X_Diff_Percent = zeros(3, 1);
MSE_Y_Diff_Percent = zeros(3, 1);
CV_Loss_X_Diff_Percent = zeros(3, 1);
CV_Loss_Y_Diff_Percent = zeros(3, 1);

% Calculate percentages
for i = 1:3
    idx = compare_indices(i);
    
    % MSE Difference (%)
    % Formula: ((Error_Other - Error_SVM) / Error_Other) * 100
    MSE_X_Diff_Percent(i) = ((mse_x_results(idx) - mse_x_results(svm_idx)) / mse_x_results(idx)) * 100;
    MSE_Y_Diff_Percent(i) = ((mse_y_results(idx) - mse_y_results(svm_idx)) / mse_y_results(idx)) * 100;
    
    % CV Loss Difference (%)
    CV_Loss_X_Diff_Percent(i) = ((cv_loss_x(idx) - cv_loss_x(svm_idx)) / cv_loss_x(idx)) * 100;
    CV_Loss_Y_Diff_Percent(i) = ((cv_loss_y(idx) - cv_loss_y(svm_idx)) / cv_loss_y(idx)) * 100;
end

% Round values for a cleaner display
MSE_X_Diff_Percent = round(MSE_X_Diff_Percent, 2);
MSE_Y_Diff_Percent = round(MSE_Y_Diff_Percent, 2);
CV_Loss_X_Diff_Percent = round(CV_Loss_X_Diff_Percent, 2);
CV_Loss_Y_Diff_Percent = round(CV_Loss_Y_Diff_Percent, 2);

% Create and display the table
SVM_Comparison_Table = table(Compared_With, MSE_X_Diff_Percent, MSE_Y_Diff_Percent, ...
    CV_Loss_X_Diff_Percent, CV_Loss_Y_Diff_Percent);

disp(SVM_Comparison_Table);
disp('=====================================================================================');
