% WildPPG Data Loader Example with Auto-Download from Hugging Face
clear all;
close all;
clc;
%%
% ---- Set data file name and Hugging Face URL ----
datafile = 'WildPPG.mat';
hf_url = 'https://huggingface.co/datasets/eth-siplab/WildPPG/resolve/main/WildPPG.mat';

% ---- Check if file exists; if not, download ----
if ~isfile(datafile)
    fprintf('Data file %s not found.\n', datafile);
    fprintf('Downloading from Hugging Face...\n');
    websave(datafile, hf_url);
    fprintf('Download complete!\n');
else
    fprintf('Found data file: %s\n', datafile);
end

% ---- Load the data ----
data = load(datafile);

% Each variable below is a cell array {16x1}, each cell = 1 subject
data_ppg_head      = data.data_ppg_head;        % PPG signals (head-worn sensor)
data_ppg_wrist     = data.data_ppg_wrist;       % PPG signals (wrist-worn sensor)
data_bpm_values    = data.data_bpm_values;      % Cleaned heart rate values
data_altitude      = data.data_altitude_values; % Altitude readings
data_temp_wrist    = data.data_temp_wrist;      % Wrist temperature readings
data_imu_chest     = data.data_imu_chest;       % IMU (chest sensor)

% ---- Example: Accessing Subject #1 ----
subject_idx = 1;
ppg_head_1   = data_ppg_head{subject_idx};
ppg_wrist_1  = data_ppg_wrist{subject_idx};
bpm_1        = data_bpm_values{subject_idx};
altitude_1   = data_altitude{subject_idx};
temp_1       = data_temp_wrist{subject_idx};
imu_chest_1  = data_imu_chest{subject_idx};

% ---- Show summary ----
fprintf('Loaded WildPPG.mat: %d subjects\n', numel(data_ppg_head));
disp('Variables loaded:');
whos('-file', datafile)
