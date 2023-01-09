import numpy as np
import matplotlib.pyplot as plt
import Peak_Detection
BIOPAC_folder_path= './../Data/2023.01.04/2023.01.04_3_gon_gu/4/'
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

    #피크 확인용

    data1_rpeak_i = np.array(
        [1222, 2744,
         5492, 7896, 10341,
         12558, 15090, 17498, 19871, 22147, 24201, 26139,
         28189, 30734,
         33199, 35551, 38315, 40977,
         43418, 45658, 48087, 50371,
         52319, 54477, 56610, 58748]
    )

    data1_rpeak_i = data1_rpeak_i.astype(int)
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i > i * size]
    data1_rpeak_i = data1_rpeak_i[data1_rpeak_i < (i + 1) * size - 1]
    plt.plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')
plt.show()

#저장용
data1_rpeak_i = np.array(
    [1222, 2744,
     5492, 7896, 10341,
     12558, 15090, 17498, 19871, 22147, 24201, 26139,
     28189, 30734,
     33199, 35551, 38315, 40977,
     43418, 45658, 48087, 50371,
     52319, 54477, 56610, 58748]
)

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

    # 피크 확인용

    data2_rpeak_i = np.array(
        [1699, 3268, 4874, 6429,
         7858, 9939, 12369, 14905, 17364,
         19441, 21723, 23910, 26255, 28901,
         31350, 33332, 35044, 36634, 37995,
         39617, 41924, 44071, 45952,
         47755, 49586, 51617, 53426, 55440,
         57387, 59277]
    )

    data2_rpeak_i = data2_rpeak_i.astype(int)
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i>i*size]
    data2_rpeak_i = data2_rpeak_i[data2_rpeak_i<(i+1)*size-1]
    plt.plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')
plt.show()

data2_rpeak_i = np.array(
    [1699, 3268, 4874, 6429,
     7858, 9939, 12369, 14905, 17364,
     19441, 21723, 23910, 26255, 28901,
     31350, 33332, 35044, 36634, 37995,
     39617, 41924, 44071, 45952,
     47755, 49586, 51617, 53426, 55440,
     57387, 59277]
)
np.save(BIOPAC_folder_path+'2022.12.27_1_gon_gu' +'_data2_rpeak_i', data2_rpeak_i)
