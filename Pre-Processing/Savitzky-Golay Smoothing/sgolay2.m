% Load your data
data = eeg_data; % Replace your_data with your actual data

% Define the optimal window size
optimal_window_size = 243
; % Assuming 701 is the optimal window size (change this to the actual optimal size)

% Initialize array to store mean log dispersion for each channel
mean_log_dispersions = zeros(size(data, 2), 1);

% Calculate mean log dispersion for each channel
for channel = 1:size(data, 2)
    % Apply Savitzky-Golay filter with the optimal window size
    smoothed_data = sgolayfilt(data(:, channel), 3, optimal_window_size);
    
    % Calculate the logarithm of the absolute differences
    log_abs_diff = log(abs(data(:, channel) - smoothed_data));
    
    % Calculate the mean log dispersion for the channel
    mean_log_dispersion = mean(log_abs_diff);
    
    % Store the mean log dispersion for the channel
    mean_log_dispersions(channel) = mean_log_dispersion;
end

% Display mean log dispersion for each channel
disp('Mean Log Dispersion for Each Channel:');
disp(mean_log_dispersions);

% Calculate and display overall mean log dispersion
overall_mean_log_dispersion = mean(mean_log_dispersions);
disp(['Overall Mean Log Dispersion: ' num2str(overall_mean_log_dispersion)]);
