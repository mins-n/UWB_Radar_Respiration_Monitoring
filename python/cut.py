import math
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io

import warnings

warnings.filterwarnings("ignore")

# Raw data extraction from .dat file ======================================
dir_path = "./../Data/2022.12.27/2022.12.27_3_sun_jin"
BIOPAC_path = "./../Data/2022.12.27/2022.12.27_3_sun_jin/2022.12.27_3_sun_jin.mat"
sample_count = 0
sample_drop_period = 434  # 해당 번째에 값은 사용 안 한다.
end_idx = 0


rawdata_path = dir_path + "/rawdata.npy"
if os.path.exists(rawdata_path):
    rawdata = np.load(rawdata_path)
else:
    for file in os.listdir(dir_path):
        if 'xethru_datafloat_' in file:
            file_path = os.path.join(dir_path, file)
            arr = np.fromfile(file_path, dtype=int)
            arr_slowindex_size = arr[2]
            arr_size = arr.size
            end_idx = 0
            start_idx = 0
            InputData = np.empty((arr_slowindex_size, 1), np.float32)
            while end_idx < arr_size:
                tmp_arr = np.fromfile(file_path, count=3, offset=end_idx * 4, dtype=np.uint32)
                id = tmp_arr[0]
                loop_cnt = tmp_arr[1]
                numCountersFromFile = tmp_arr[2]
                start_idx = end_idx + 3
                end_idx += 3 + numCountersFromFile
                fInputData = np.fromfile(file_path, count=numCountersFromFile, offset=start_idx * 4, dtype=np.float32)
                sample_count += 1
                if sample_count % sample_drop_period == 0:
                    continue
                fInputData = np.array(fInputData).reshape(numCountersFromFile, 1)
                InputData = np.append(InputData, fInputData, axis=1)  # Raw data
    rawdata = np.array(InputData[:, 1:], dtype=np.double)
    np.save(rawdata_path, rawdata)

fast_to_m = 0.006445  # fast index to meter
UWB_Radar_index_start = 0.5  # UWB Radar Range 0.5 ~ 2.5m
UWB_Radar_index_start = math.floor(UWB_Radar_index_start / fast_to_m)

Window_rawdata = np.array(rawdata[:, 3000:3600])
SD = np.array([])
for i in range(len(Window_rawdata)):
    SD = np.append(SD, np.std(Window_rawdata[i]))  # 거리에 대한 표준편차 배열
Max = max(SD)  # 가장 큰 표준편차 값
Index = np.argmax(SD)  # 가장 큰 표준편차 Idx : MAX 위치 값
Pm = np.mean(SD[Index - 2:Index + 1])  # 가장 높은 표쥰 편차와 -1, +1 idx의 평균값
Index = Index + UWB_Radar_index_start

d0 = Index * fast_to_m  # 가장 높은 편차의 거리 meter
n = np.mean(SD)  # 편차 배열의 평균

baselineThreashold = (Pm - n) / (2 * d0 + 1) + n

# Dynamic Threshold ===========================================================================
di = np.arange(1, len(rawdata) + 1, 1)
di = di + UWB_Radar_index_start
di = di * fast_to_m

k = np.array([])
for i in range(len(di)):
    k = np.append(k, di[i] ** 2 / d0 ** 2)
k = k ** (-1)
Dynamic_threshold = np.array(k) * baselineThreashold

TC_matrix = np.array([])
Distance = np.zeros((0, 2))

TC_matrix = SD >= Dynamic_threshold

for i in range(len(TC_matrix)):
    if TC_matrix[i] == 0 and (i > 2) and (i < len(TC_matrix) - 1):
        if TC_matrix[i - 1] == 1 and TC_matrix[i + 1] == 1:
            TC_matrix[i] = 1

TC_cnt = 0
Human_cnt = 0

for i in range(len(rawdata)):
    if TC_matrix[i]:
        TC_cnt += 1
    else:
        if TC_cnt < 20:
            TC_matrix[i - TC_cnt: i] = 0
            TC_cnt = 0
        elif TC_cnt > 75:
            TC_matrix[i - TC_cnt: i] = 0
            TC_cnt = 0
        else:
            Human_cnt += 1
            Distance = np.r_[Distance, [[0, 0]]]
            Distance[Human_cnt - 1, :] = [i - TC_cnt, i - 1]
            TC_cnt = 0
if TC_cnt != 0:
    if TC_cnt < 20:
        TC_matrix[i - TC_cnt:] = 0
        TC_cnt = 0
    elif TC_cnt > 75:
        TC_matrix[i - TC_cnt:] = 0
        TC_cnt = 0
    else:
        Human_cnt += 1
        Distance = np.r_[Distance, [[0, 0]]]
        Distance[Human_cnt - 1, :] = [i - TC_cnt, i - 1]
        TC_cnt = 0

Max_sub = np.zeros((Human_cnt, 1))
Max_sub_Index = np.zeros((Human_cnt, 1))
for i in range(Human_cnt):
    Max_sub[i, 0] = max(SD[int(Distance[i, 0]) - 1:int(Distance[i, 1])])
    Max_sub_Index[i, 0] = np.argmax(SD[int(Distance[i, 0]) - 1:int(Distance[i, 1])])

    Max_sub_Index[i, 0] += Distance[i, 0]
    for i in range(Human_cnt):
        Max_sub[i, 0] = max(SD[int(Distance[i, 0]) - 1:int(Distance[i, 1])])
        Max_sub_Index[i, 0] = np.argmax(SD[int(Distance[i, 0]) - 1:int(Distance[i, 1])])

        Max_sub_Index[i, 0] += Distance[i, 0]

        if len(rawdata) < Max_sub_Index[i, 0] + 15:
            Distance[i, 0] = Max_sub_Index[i, 0] - 15
            Distance[i, 1] = len(rawdata[1])
        elif Max_sub_Index[i, 0] - 15 < 1:
            Distance[i, 0] = 1
            Distance[i, 1] = Max_sub_Index[i, 0] + 15
        else:
            Distance[i, 0] = Max_sub_Index[i, 0] - 15
            Distance[i, 1] = Max_sub_Index[i, 0] + 15

Data = TC_matrix.reshape(TC_matrix.size, 1) * rawdata
fs = 20
show_idx = 60 # 데이터들의 시작부터 몇초 볼껀지
# Print the detected peaks
for i in range(Human_cnt):
    UWB_data = rawdata[Max_sub_Index[i, 0].astype(int)][:fs * show_idx]
    plt.figure(num=1, figsize=(10, 8))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
    plt.subplot(Human_cnt + 1, 1, 1 + i)
    plt.subplot(Human_cnt + 1, 1, 1 + i).set_title("UWB Peak Detection " + "Human " + str(i + 1))
    plt.plot(UWB_data)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

data = scipy.io.loadmat(BIOPAC_path)
data = np.array(data['channels'])  # Convert the data to a NumPy array

data1 = data[0][0][0][0][0][0]
data1_fs = int(data[0][0][0][0][1][0])
data1 = data1.flatten()  # Flatten the data if it is not 1D
data1 = data1.astype(float)  # Cast to float if necessary

data2 = data[0][1][0][0][0][0]
data2_fs = int(data[0][1][0][0][1][0])
data2 = data2.flatten()  # Flatten the data if it is not 1D
data2 = data2.astype(float)  # Cast to float if necessary

plt.figure(num=2, figsize=(10, 8))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
plt.subplot(3, 1, 1)
plt.subplot(3, 1, 1).set_title("BIOPAC Peak Detection Data1")
plt.plot(data1[:data1_fs*show_idx])
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.subplot(3, 1, 2).set_title("BIOPAC Peak Detection Data2")
plt.plot(data2[:data2_fs*show_idx])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

cut_idx = int(input("UWB Rawdata 그래프를 참고하여 자를 부분에 index를 입력하시오 "))
UWB_cut_path = dir_path + "/UWB_cut.npy"
BIOPAC_cut_path = dir_path + "/BIOPAC_cut.npy"

BIOPAC_cut = []
UWB_cut = rawdata[:,cut_idx:]
BIOPAC_cut.append(data1[int(cut_idx*data1_fs/fs):])
BIOPAC_cut.append(data1_fs)
BIOPAC_cut.append(data2[int(cut_idx*data2_fs/fs):])
BIOPAC_cut.append(data2_fs)

np.save(UWB_cut_path,UWB_cut)
np.save(BIOPAC_cut_path, BIOPAC_cut)