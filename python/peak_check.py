import numpy as np
from matplotlib import pyplot as plt

base_path = "./../Data/2023.01.02/2023.01.02_2_sun_chan"
base_path = base_path + "/"


BIOPAC_folder_path = base_path + str(0) + "/"
BIOPAC_path = BIOPAC_folder_path+"BIOPAC_data.npy"
data1_rpeak_i_path = BIOPAC_folder_path+ "data1_rpeak_i.npy"
data2_rpeak_i_path = BIOPAC_folder_path+"data2_rpeak_i.npy"

data1_rpeak = np.load(data1_rpeak_i_path)
data2_rpeak = np.load(data2_rpeak_i_path)

data = np.load(BIOPAC_path, allow_pickle=True)
data1 = data[0] # 1번째 사람 데이터
data1_fs = data[1] # 바이오팩 fs

data2 = data[2]
data2_fs = data[3]

length = 5
size = len(data1)//length
for i in range(length):
    plt.figure(num=1, figsize=(12, 7))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace = 1)
    plt.subplot(length, 1, i+1)
    plt.subplot(length, 1, i+1).set_title("BIOPAC Peak Detection Data1")

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.xlim([i*size , (i+1)*size-1])
    plt.plot(data1)

    data1_rpeak_i = data1_rpeak

    data1_rpeak_i=data1_rpeak_i.astype(int)
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i>i*size]
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i<(i+1)*size-1]
    plt.plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')


size = len(data2)//length
for i in range(length):
    plt.figure(num=2, figsize=(12,7))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace =1)
    plt.subplot(length, 1, i+1)
    plt.subplot(length, 1, i+1).set_title("BIOPAC Peak Detection Data2")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.xlim([i*size , (i+1)*size-1])
    plt.plot(data2)

    data2_rpeak_i = data2_rpeak

    data2_rpeak_i = data2_rpeak_i.astype(int)
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i>i*size]
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i<(i+1)*size-1]
    plt.plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')

plt.show()