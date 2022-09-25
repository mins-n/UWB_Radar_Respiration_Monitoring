%% Raw data extraction from .dat file

clear,clc,close all;


cd('./Data/0830_광진,윤곤/0830_3rd_광진(1m)_윤곤(1.5m)');

% Load UWB data
FileList = dir('xethru_datafloat_*.dat');
FileName = {FileList.name}';
fid_raw = fopen(FileName{1},'r');

cd('..\..\..\');

sample_count = 0;
sample_drop_period = 434; %% 이 이상 거리는 사용 안한다
DataCursor = 1;
rawdata = [];
InputData = [];

while (1)
    id = fread(fid_raw,1,'uint32');
    if length(id) < 1, break; end
    loop_cnt = fread(fid_raw,1,'uint32');
    numCountersFromFile = fread(fid_raw,1,'uint32');
    fInputData = fread(fid_raw,[1 numCountersFromFile],'single');
    sample_count = sample_count + 1;

    if mod(sample_count, sample_drop_period) == 0
        continue;
    end

    InputData(:,DataCursor)= double(fInputData); % Raw data
    DataCursor = DataCursor + 1;

end
rawdata = [rawdata InputData];
fid_raw = fclose(fid_raw);
save rawdata.mat rawdata



