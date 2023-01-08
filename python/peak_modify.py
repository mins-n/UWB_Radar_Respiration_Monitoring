import math
import numpy as np
import matplotlib.pyplot as plt
import os
import Peak_Detection
import scipy.io
BIOPAC_folder_path= './../Data/2022.12.27/2022.12.27_1_gon_gu/4/'
BIOPAC_path = BIOPAC_folder_path+"BIOPAC_data.npy" 


BIOPAC_rpeak_i = []

data = np.load(BIOPAC_path, allow_pickle=True)
#data = np.array(data['channels'])  # Convert the data to a NumPy array

data1 = data[0] #1번째 사람 데이터
data1_fs = data[1] #바이오팩 fs

data2 = data[2]
data2_fs = data[3]

# Set the sample rate and minimum interval
MI = 1  # Minimum interval in seconds

data1_rpeak_i, data1_env_peak_i = Peak_Detection.Peak_Detection(data1, data1_fs, MI) #peak detection
BIOPAC_rpeak_i.append(data1_rpeak_i)

data2_rpeak_i, data2_env_peak_i = Peak_Detection.Peak_Detection(data2, data2_fs, MI)
BIOPAC_rpeak_i.append(data2_rpeak_i)

# Print the detected peaks
print(data1_rpeak_i)
print(data2_rpeak_i)

length = 5
size = len(data1)//length
for i in range(length):
    plt.figure(num=2, figsize=(10, 300))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace = 1)
    plt.subplot(length, 1, i+1)
    plt.subplot(length, 1, i+1).set_title("BIOPAC Peak Detection Data1")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.xlim([i*size , (i+1)*size-1])
    plt.plot(data1)

    data1_rpeak_i, data1_env_peak_i = Peak_Detection.Peak_Detection(data1, data1_fs, MI)#peak 적용

    #피크 확인용
    
    data1_rpeak_i = np.array(
        [  5792,  14118,  23028,  30809,  36597,  45176,  54421,  61498,
  67215,  74148,  82003,  90566, 99873, 108465, 116482,
 123897, 130973, 137065, 143483, 150367, 157484, 164853, 172617, 178985,
 185925, 193810, 199577, 206449, 212093, 217601, 223519, 229245, 234873])
    data1_rpeak_i=data1_rpeak_i.astype(int)
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i>i*size]
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i<(i+1)*size-1]
    plt.plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')
plt.show()

#저장용
data1_rpeak_i = np.array(
        [  5792,  14118,  23028,  30809,  36597,  45176,  54421,  61498,
  67215,  74148,  82003,  90566, 99873, 108465, 116482,
 123897, 130973, 137065, 143483, 150367, 157484, 164853, 172617, 178985,
 185925, 193810, 199577, 206449, 212093, 217601, 223519, 229245, 234873])

np.save(BIOPAC_folder_path+'2022.12.27_1_gon_gu'+'_data1_rpeak_i', data1_rpeak_i)


size = len(data2)//length
for i in range(length):
    plt.figure(num=2, figsize=(10, 300))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace = 1)
    plt.subplot(length, 1, i+1)
    plt.subplot(length, 1, i+1).set_title("BIOPAC Peak Detection Data2")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.xlim([i*size , (i+1)*size-1])
    plt.plot(data2)
    data2_rpeak_i, data1_env_peak_i = Peak_Detection.Peak_Detection(data2, data2_fs, MI)
    
    data2_rpeak_i = np.array([7668,  18748,  30700,  41680,  53052,  64087,  73650,  84170, 93890, 103095,
 111956, 121413, 130519, 139528, 148201, 157789, 168006, 178204, 188761,
 198558, 208260, 216461, 224838, 232940])

    data2_rpeak_i = data2_rpeak_i.astype(int)
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i>i*size]
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i<(i+1)*size-1]
    plt.plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')
plt.show()


data2_rpeak_i = np.array([7668,  18748,  30700,  41680,  53052,  64087,  73650,  84170, 93890, 103095,
 111956, 121413, 130519, 139528, 148201, 157789, 168006, 178204, 188761,
 198558, 208260, 216461, 224838, 232940])
np.save(BIOPAC_folder_path+'2022.12.27_1_gon_gu' +'_data2_rpeak_i', data2_rpeak_i)
