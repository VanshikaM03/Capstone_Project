% Assuming your EEG data is in a variable called eegData
% Size of eegData is [T, 59] where T is the number of time points

% Z-score standardization
standardizedEEG = zscore(smoothed_data);

% The zscore function in MATLAB standardizes each column by default.
% It calculates the mean and standard deviation for each column and applies:
% standardizedEEG(:, i) = (eegData(:, i) - mean(eegData(:, i))) / std(eegData(:, i))