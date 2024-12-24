\% Load your data and select one channel
data = eeg_data; % Replace eeg_data with your actual data
channel = 29; % Select the channel to apply the filter

% Define the range of window sizes (odd sizes from 101 to 999)
window_sizes = 101:2:999;
num_sizes = length(window_sizes);

% Initialize arrays to store log dispersion values
log_dispersions = zeros(num_sizes, 1);

% Apply Savitzky-Golay filter for each window size and calculate log dispersion
for i = 1:num_sizes
    window_size = window_sizes(i);
    smoothed_data = sgolayfilt(data(:, channel), 3, window_size);
    log_dispersions(i) = std(log(abs(data(:, channel) - smoothed_data)));
end

% Find the optimal window size
[~, optimal_index] = min(log_dispersions);
optimal_window_size = window_sizes(optimal_index);


% Apply Savitzky-Golay filter with the optimal window size
optimal_smoothed_data = sgolayfilt(data(:, channel), 3, optimal_window_size);
optimal_window_size