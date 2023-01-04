import math
import numpy as np
import matplotlib.pyplot as plt
import os
import Peak_Detection
import scipy.io

BIOPAC_path = "./../Data/2022.12.27/2022.12.27_3_sun_jin/4/BIOPAC_data.npy"
BIOPAC_folder_path= './../Data/2022.12.27/2022.12.27_3_sun_jin/4/'

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
        [5415, 19035, 31710, 45393, 58378, 72433, 83966, 92325, 101967,
         111831, 122803, 133387, 144549, 157146, 169359, 181066, 193205,
         201350, 210300, 219070, 230280, 236188])
    data1_rpeak_i=data1_rpeak_i.astype(int)
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i>i*size]
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i<(i+1)*size-1]
    plt.plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')
plt.show()

#저장용
"""
data1_rpeak_i = np.array(
    [5415, 19035, 31710, 45393, 58378, 72433, 83966, 92325, 101967,
         111831, 122803, 133387, 144549, 157146, 169359, 181066, 193205,
         201350, 210300, 219070, 230280, 236188])

np.save(BIOPAC_folder_path+'2022.12.27_3_sun_jin_data1_rpeak_i', data1_rpeak_i)
"""

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

    data2_rpeak_i = np.array(
        [8285, 15641, 23150, 31460, 38852, 45558, 52213, 58863, 65886, 72719,
         79637, 85787, 91480, 96742, 102277, 107821, 114049, 120272, 126290, 132241,
         138787, 145188, 151490, 158291, 164928, 171572, 177749, 184087, 190430, 200000,
         204770, 209593, 215057, 220321, 227089, 232618, 237604,
         0, ])

    data2_rpeak_i = data2_rpeak_i.astype(int)
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i>i*size]
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i<(i+1)*size-1]
    plt.plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')
plt.show()

"""
data2_rpeak_i = np.array(
        [8285, 15641, 23150, 31460, 38852, 45558, 52213, 58863, 65886, 72719,
         79637, 85787, 91480, 96742, 102277, 107821, 114049, 120272, 126290, 132241,
         138787, 145188, 151490, 158291, 164928, 171572, 177749, 184087, 190430, 200000,
         204770, 209593, 215057, 220321, 227089, 232618, 237604,
         0, ])
np.save(BIOPAC_folder_path+'2022.12.27_3_sun_jin_data2_rpeak_i', data2_rpeak_i)
"""