import numpy as np
import matplotlib.pyplot as plt
import Peak_Detection

BIOPAC_folder_path= './../Data/2023.01.18/2023.01.18_2_goo_gon/4/'
BIOPAC_path = BIOPAC_folder_path+"BIOPAC_data.npy" 


BIOPAC_rpeak_i = []

data = np.load(BIOPAC_path, allow_pickle=True)

data1 = data[0] # 1번째 사람 데이터
data1_fs = data[1] # 바이오팩 fs

data2 = data[2]
data2_fs = data[3]

# Set the sample rate and minimum interval
MI = 1  # Minimum interval in seconds

data1_rpeak_i, data1_env_peak_i = Peak_Detection.Peak_Detection(data1, data1_fs, MI) #peak detection
data1_rpeak_i  = data1_rpeak_i.astype(int)
BIOPAC_rpeak_i.append(data1_rpeak_i)

data2_rpeak_i, data2_env_peak_i = Peak_Detection.Peak_Detection(data2, data2_fs, MI)
data2_rpeak_i = data2_rpeak_i.astype(int)
BIOPAC_rpeak_i.append(data2_rpeak_i)

# Print the detected peaks
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

length = 5
size = len(data1)//length
for i in range(length):
    plt.figure(num=1, figsize=(10, 300))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace = 1)
    plt.subplot(length, 1, i+1)
    plt.subplot(length, 1, i+1).set_title("BIOPAC Peak Detection Data1")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.xlim([i*size , (i+1)*size-1])
    plt.plot(data1)

    data1_rpeak_i, data1_env_peak_i = Peak_Detection.Peak_Detection(data1, data1_fs, MI)#peak 적용

    # 피크 확인
    ###
    data1_rpeak_i = np.array(
        [1434, 4810, 8057, 11679, 15274,
         18435, 21397, 24484, 27416, 30407,
         33272, 36384, 39435, 42483,
         45408, 48585, 52071, 55349,
         58871]
    )
    ###

    data1_rpeak_i=data1_rpeak_i.astype(int)
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i>i*size]
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i<(i+1)*size-1]
    plt.plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')
plt.show()


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

    # 피크 확인
    ###
    data2_rpeak_i = np.array(
        [1194, 3778, 6875, 9556,
         12078, 15625, 18178, 21273,
         24083, 27403, 32020,
         34849, 40565, 44975,
         48195, 51107,
         53914, 56499, 59281,
         ]

    )
    ###

    data2_rpeak_i = data2_rpeak_i.astype(int)
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i>i*size]
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i<(i+1)*size-1]
    plt.plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')
plt.show()


#저장

np.save(BIOPAC_folder_path+'data1_rpeak_i', data1_rpeak_i)
np.save(BIOPAC_folder_path +'data2_rpeak_i', data2_rpeak_i)
