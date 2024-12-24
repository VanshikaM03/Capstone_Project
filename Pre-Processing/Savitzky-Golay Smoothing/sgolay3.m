% Assuming eeg_data is a matrix of size [num_samples, 59] where each column represents a channel
[num_samples, num_channels] = size(eeg_data);
sampling_rate = 100;  % Assuming a sampling rate of 1000 samples per second
time = (0:num_samples-1) / sampling_rate;  % Time vector in seconds

% Applying Savitzky-Golay filtering to each channel
window_size = 107;
polynomial_order = 3;
smoothed_data = sgolayfilt(eeg_data, polynomial_order, window_size);

% Plotting the original and smoothed data for one channel (e.g., channel 1)
channel_to_plot = 1;
figure;
%plot(time, eeg_data(:, channel_to_plot), 'b', 'LineWidth', 1.5);
%hold on;
plot(time, smoothed_data(:, channel_to_plot), 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original and Smoothed EEG Data for Channel 1');
legend('Original', 'Smoothed');
xlim([0, 500]);