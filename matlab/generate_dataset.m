% Generate CSI dataset for 5G TDL channel with SRS periodicity
% Requires 5G Toolbox

clear; clc;

%% Environment parameters
fc = 28e9;                 % Carrier frequency (Hz)
ueSpeedKmh = 60;           % UE speed (km/h)
ueSpeed = ueSpeedKmh/3.6;  % (m/s)
c = 3e8;
maxDoppler = (ueSpeed/c) * fc;

srsPeriod = 0.000625; % 0.625 ms
numSteps = 2000;      % number of SRS snapshots
numUE = 4;            % number of UEs

%% Carrier configuration (FR2)
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 120;
carrier.NSizeGrid = 66; % 66 RBs (~100 MHz)

ofdmInfo = nrOFDMInfo(carrier);

%% SRS configuration
srs = nrSRSConfig;
srs.NumSRSPorts = 8;
srs.NumSRSSymbols = 1;
srs.SymbolStart = 0;
srs.CSRS = 0;
srs.KTC = 2;

srsInd = nrSRSIndices(carrier, srs);
srsSym = nrSRS(carrier, srs);

%% Prepare a single SRS waveform (for one snapshot)
txGrid = nrResourceGrid(carrier, srs.NumSRSPorts);
txGrid(srsInd) = srsSym;
txWave = nrOFDMModulate(carrier, txGrid);

numSubcarriers = carrier.NSizeGrid * 12;
numTx = srs.NumSRSPorts;

%% Allocate dataset
H = complex(zeros(numSteps, numUE, numSubcarriers, numTx));
time = (0:numSteps-1).' * srsPeriod;

%% Channel generation per UE
for ue = 1:numUE
    tdl = nrTDLChannel;
    tdl.DelayProfile = 'TDL-C';
    tdl.DelaySpread = 30e-9;
    tdl.MaximumDopplerShift = maxDoppler;
    tdl.NumTransmitAntennas = numTx;
    tdl.NumReceiveAntennas = 1;
    tdl.SampleRate = ofdmInfo.SampleRate;
    tdl.NormalizePathGains = true;
    pathFilters = getPathFilters(tdl);

    for n = 1:numSteps
        release(tdl);
        tdl.InitialTime = (n-1) * srsPeriod;
        [~, pathGains, ~] = tdl(txWave);
        Hest = nrPerfectChannelEstimate(carrier, pathGains, pathFilters);
        % Hest dims: [subcarriers x symbols x rx x tx]
        Hsnap = squeeze(Hest(:,1,1,:)); % [subcarriers x tx]
        H(n, ue, :, :) = Hsnap;
    end
end

%% Save dataset
params.fc = fc;
params.ueSpeedKmh = ueSpeedKmh;
params.maxDoppler = maxDoppler;
params.srsPeriod = srsPeriod;
params.numSteps = numSteps;
params.numUE = numUE;
params.numTx = numTx;
params.numSubcarriers = numSubcarriers;
params.subcarrierSpacing = carrier.SubcarrierSpacing;
params.nSizeGrid = carrier.NSizeGrid;

outPath = fullfile('..','data','csi_dataset.mat');
save(outPath, 'H', 'time', 'params', '-v7.3');

fprintf('Saved dataset to %s\n', outPath);
