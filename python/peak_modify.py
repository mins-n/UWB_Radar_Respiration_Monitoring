import math
import numpy as np
import matplotlib.pyplot as plt
import os
import Peak_Detection
import scipy.io
BIOPAC_folder_path= './../Data/2022.12.26/2022.12.26_4_soo_jin/3/'
BIOPAC_path = BIOPAC_folder_path+"BIOPAC_data.npy" 


BIOPAC_rpeak_i = []

data = np.load(BIOPAC_path, allow_pickle=True)

data1 = data[0]
data1_fs = data[1]

data2 = data[2]
data2_fs = data[3]

MI = 1

data1_rpeak_i, data1_env_peak_i = Peak_Detection.Peak_Detection(data1, data1_fs, MI)
data1_rpeak_i  = data1_rpeak_i.astype(int)
BIOPAC_rpeak_i.append(data1_rpeak_i)

data2_rpeak_i, data2_env_peak_i = Peak_Detection.Peak_Detection(data2, data2_fs, MI)
data2_rpeak_i = data2_rpeak_i.astype(int)
BIOPAC_rpeak_i.append(data2_rpeak_i)


print("[",end="")
for i in range(int(len(data1_rpeak_i[:-1])/5) + 1):
    if(5*(i+1) > len(data1_rpeak_i[:-1])):
        print(*data1_rpeak_i[5*i:-1].tolist(), sep=', ', end="] \n\n")
    else:
	    print(*data1_rpeak_i[5*i:5*(i+1)].tolist(), sep=', ', end = ',\n')
print("[",end="")
for i in range(int(len(data2_rpeak_i[:-1])/5) + 1):
    if(5*(i+1) > len(data2_rpeak_i[:-1])):
        print(*data2_rpeak_i[5*i:-1].tolist(), sep=', ', end="] \n\n")
    else:
	    print(*data2_rpeak_i[5*i:5*(i+1)].tolist(), sep=', ', end = ',\n')

length = 4
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

    data1_rpeak_i, data1_env_peak_i = Peak_Detection.Peak_Detection(data1, data1_fs, MI)

    #
    data1_rpeak_i = np.array(
[5727, 13780, 21582, 29745, 38226,
45967, 53938, 60910, 68558, 75523, 82622,
89566, 96940, 103791, 110990, 117849,
125140, 131580, 138685, 146011, 153225,
161490, 169473, 176392, 184607, 192697,
201205, 209634, 217045, 224995, 232689,
] 
     )
    #
    data1_rpeak_i=data1_rpeak_i.astype(int)
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i>i*size]
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i<(i+1)*size-1]
    plt.plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')
plt.show()


#
data1_rpeak_i = np.array(
[5727, 13780, 21582, 29745, 38226,
45967, 53938, 60910, 68558, 75523, 82622,
89566, 96940, 103791, 110990, 117849,
125140, 131580, 138685, 146011, 153225,
161490, 169473, 176392, 184607, 192697,
201205, 209634, 217045, 224995, 232689,
] 
     )
np.save(BIOPAC_folder_path +'data1_rpeak_i', data1_rpeak_i)
#

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
    
    #
    data2_rpeak_i = np.array(
    [1535, 9338, 17340, 25560, 32510, 40548, 47540,
54139, 60213, 66998, 73539, 79465,
86040, 93150, 99050, 105245, 111636, 118114, 124508,
131330, 137784, 144500, 151877, 159051,
166092, 172951, 180047, 187545, 195140,
202020, 208717, 214958, 221805, 228826,
236520] 
    )
    #
    data2_rpeak_i = data2_rpeak_i.astype(int)
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i>i*size]
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i<(i+1)*size-1]
    plt.plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')
plt.show()

#
data2_rpeak_i = np.array(
        [1535, 9338, 17340, 25560, 32510, 40548, 47540,
54139, 60213, 66998, 73539, 79465,
86040, 93150, 99050, 105245, 111636, 118114, 124508,
131330, 137784, 144500, 151877, 159051,
166092, 172951, 180047, 187545, 195140,
202020, 208717, 214958, 221805, 228826,
236520] 
    )

np.save(BIOPAC_folder_path +'data2_rpeak_i', data2_rpeak_i)
#