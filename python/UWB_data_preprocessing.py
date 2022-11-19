# Raw data extraction from .dat file
import numpy as np
import os

dir_path = "./../Data/0927_광진,구찬/0927_1st_구찬(90cm)_광진(1m 90)"
sample_count = 0
sample_drop_period = 434  # 해당 번째에 값은 사용 안 한다.
end_idx = 0

for file in os.listdir(dir_path):
    if 'xethru_datafloat_' in file:
        file_path = os.path.join(dir_path, file)
        arr = np.fromfile(file_path, dtype=int)
        arr_slowindex_size = arr[2]
        arr_size = arr.size
        end_idx = 0
        start_idx = 0
        InputData = np.empty((arr_slowindex_size,1), np.float32)
        while end_idx < arr_size:
            tmp_arr = np.fromfile(file_path, count=3,offset=end_idx*4, dtype=np.uint32)
            id = tmp_arr[0]
            loop_cnt = tmp_arr[1]
            numCountersFromFile = tmp_arr[2]
            start_idx = end_idx + 3
            end_idx += 3 + numCountersFromFile
            fInputData = np.fromfile(file_path, count=numCountersFromFile, offset=start_idx*4, dtype=np.float32)
            sample_count += 1
            if sample_count % sample_drop_period == 0:
                continue
            fInputData = np.array(fInputData).reshape(numCountersFromFile,1)
            InputData = np.append(InputData,fInputData,axis=1)  # Raw data
rawdata = np.array(InputData[:,1:],dtype=np.double)