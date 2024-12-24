% Assuming standardizedEEG is the z-score standardized data with size [189802, 59]

% Set desired range for min-max normalization
minVal = 0;
maxVal = 1;

% Initialize a matrix to store the normalized data
normalizedEEG = zeros(size(standardizedEEG));

% Apply min-max normalization for each channel (column)
for ch = 1:size(standardizedEEG, 2)
    % Get the minimum and maximum of the current channel
    minChannel = min(standardizedEEG(:, ch));
    maxChannel = max(standardizedEEG(:, ch));
    
    % Apply min-max normalization to scale data to [minVal, maxVal]
    normalizedEEG(:, ch) = (standardizedEEG(:, ch) - minChannel) / (maxChannel - minChannel) * (maxVal - minVal) + minVal;
end