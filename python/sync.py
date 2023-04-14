import math
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.io
import util
from tqdm import tqdm
import sys

import warnings
warnings.filterwarnings("ignore")

def get_data_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith(".acq")):
                root = root.replace("\\", "/")
                file_list.append(root)
    return file_list

root_dir = "./../Data/"
file_list = get_data_path(root_dir)

for dir_path in tqdm(file_list):
    file_name = os.path.basename(dir_path)
    BIOPAC_path = dir_path + "/" + file_name + ".mat"
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
    show = False # True: plt 출력, False: plt 출력 안함

    Window_rawdata = np.array(rawdata[:, 800:2400])
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
    Human = 2

    dynamic_TC = 16
    while Human_cnt < 2:
        # print("Human_cnt:%d Dynamic_TC:%d" % (Human_cnt, dynamic_TC))
        if dynamic_TC == 1: break
        Human_cnt = 0
        dynamic_TC -= 1
        TC_cnt = 0

        TC_matrix = SD >= Dynamic_threshold

        for i in range(len(TC_matrix)):
            if TC_matrix[i] == 0 and (i > 2) and (i < len(TC_matrix) - 1):
                if TC_matrix[i - 1] == 1 and TC_matrix[i + 1] == 1:
                    TC_matrix[i] = 1

        for i in range(len(rawdata)):
            if TC_matrix[i]:
                TC_cnt += 1
            else:
                if TC_cnt < dynamic_TC:
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
            if TC_cnt < dynamic_TC:
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
    Data = TC_matrix.reshape(TC_matrix.size, 1) * rawdata

    if Human_cnt < Human:
        print("======================================================================")
        print(dir_path)
        print("======================================================================")
        sys.exit()


    fs = 20
    MI = 1
    UWB_rpeak_i = []
    # Print the detected peaks
    for i in range(Human_cnt):
        UWB_data = rawdata[Max_sub_Index[i, 0].astype(int),:]
        rpeak_i, env_peak_i = util.Peak_Detection(UWB_data, fs, MI)
        UWB_rpeak_i.append(rpeak_i)
        if show == True:
            plt.figure(num=1, figsize=(10, 8))
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
            plt.subplot(Human_cnt, 1, 1 + i)
            plt.subplot(Human_cnt, 1, 1 + i).set_title("UWB Peak Detection " + "Human " + str(i + 1))
            plt.plot(UWB_data)
            rpeak_i = rpeak_i.astype(int)
            plt.plot(rpeak_i, UWB_data[rpeak_i], 'ro')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
    if show == True:
        print(UWB_rpeak_i[0])
        print(UWB_rpeak_i[1])


    BIOPAC_rpeak_i = []

    data = scipy.io.loadmat(BIOPAC_path)
    data = np.array(data['channels'])  # Convert the data to a NumPy array

    data1_origin = data[0][0][0][0][0][0]
    data2_origin = data[0][1][0][0][0][0]
    biopac_fs = int(data[0][0][0][0][1][0])
    sampling = int(biopac_fs / fs)

    i=0
    data1 = np.array([])
    while(i<=len(data1_origin)):
        data1 = np.append(data1,data1_origin[i])
        i += sampling
    i=0
    data2 = np.array([])
    while(i<=len(data2_origin)):
        data2 = np.append(data2, data2_origin[i])
        i += sampling

    data1_fs = fs
    data1 = data1.flatten()  # Flatten the data if it is not 1D
    data1 = data1.astype(float)  # Cast to float if necessary

    data2_fs = fs
    data2 = data2.flatten()  # Flatten the data if it is not 1D
    data2 = data2.astype(float)  # Cast to float if necessary

    # Set the sample rate and minimum interval
    MI = 1  # Minimum interval in seconds

    data1_rpeak_i, data1_env_peak_i = util.Peak_Detection(data1, data1_fs, MI)
    data2_rpeak_i, data2_env_peak_i = util.Peak_Detection(data2, data2_fs, MI)

    BIOPAC_rpeak_i.append(data1_rpeak_i)
    BIOPAC_rpeak_i.append(data2_rpeak_i)

    data1_rpeak_i = data1_rpeak_i.astype(int)
    data2_rpeak_i = data2_rpeak_i.astype(int)
    if show == True:
        # Print the detected peaks
        print(data1_rpeak_i)
        print(data2_rpeak_i)

        plt.figure(num=2, figsize=(10, 20))
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
        plt.subplot(2, 1, 1)
        plt.subplot(2, 1, 1).set_title("BIOPAC Peak Detection Data1")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.plot(data1)
        plt.plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')

        plt.subplot(2, 1, 2)
        plt.subplot(2, 1, 2).set_title("BIOPAC Peak Detection Data2")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.plot(data2)
        plt.plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')
        plt.show()

    diff = 9999
    for i in range(0,len(UWB_rpeak_i[0])-5) :
        for j in range(0,len(data1_rpeak_i)-5) :
            tmp = 0
            for k in range(0, 4):
                tmp += abs(UWB_rpeak_i[0][i+k]-data1_rpeak_i[j+k])
            if tmp<diff and abs(UWB_rpeak_i[0][i]-data1_rpeak_i[j])<40:
                diff = tmp
                start_idx1 = UWB_rpeak_i[0][i]
                start_idx2 = data1_rpeak_i[j]

    sync = abs(int(start_idx1 - start_idx2))

    UWB_sync_path = dir_path + "/UWB_sync.npy"
    BIOPAC_sync_path = dir_path + "/BIOPAC_sync.npy"

    show = 1200

    UWB_sync = rawdata[:, int(sync):]
    if show == True:
        plt.subplot(4, 1, 1)
        plt.plot(UWB_sync[Max_sub_Index[0, 0].astype(int), show:show + 500])
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.subplot(4, 1, 2)
        plt.plot(data1_origin[show * sampling:(show + 500) * sampling])

        im = UWB_sync[Max_sub_Index[0, 0].astype(int) - 15:Max_sub_Index[0, 0].astype(int) + 15, show:show + 500]
        L = 0.02
        [u, ux, uy] = util.l0_grad_minimization(im, L)
        Scharr_dx = cv2.Scharr(u.astype('float32'), -1, 1, 0)
        Scharr_dy = cv2.Scharr(u.astype('float32'), -1, 0, 1)
        Scharr_mag = cv2.magnitude(Scharr_dx, Scharr_dy)
        img = cv2.bilateralFilter(Scharr_mag, -1, 5, 10)
        plt.subplot(4, 1, 3)
        plt.pcolor(img)
        plt.subplot(4, 1, 4)
        plt.imshow(img, cmap='gray')
        plt.show()

        plt.subplot(4, 1, 1)
        plt.plot(UWB_sync[Max_sub_Index[1, 0].astype(int), show:show + 500])
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.subplot(4, 1, 2)
        plt.plot(data2_origin[show * sampling:(show + 500) * sampling])
        im = UWB_sync[Max_sub_Index[1, 0].astype(int) - 15:Max_sub_Index[1, 0].astype(int) + 15, show:show + 500]
        L = 0.02
        [u, ux, uy] = util.l0_grad_minimization(im, L)
        Scharr_dx = cv2.Scharr(u.astype('float32'), -1, 1, 0)
        Scharr_dy = cv2.Scharr(u.astype('float32'), -1, 0, 1)
        Scharr_mag = cv2.magnitude(Scharr_dx, Scharr_dy)
        img = cv2.bilateralFilter(Scharr_mag, -1, 5, 10)
        plt.subplot(4, 1, 3)
        plt.pcolor(img)
        plt.subplot(4, 1, 4)
        plt.imshow(img, cmap='gray')
        plt.show()

    BIOPAC_sync = []
    BIOPAC_sync.append(data1_origin)
    BIOPAC_sync.append(biopac_fs)

    BIOPAC_sync.append(data2_origin)
    BIOPAC_sync.append(biopac_fs)

    print() # 보기 편하게 줄바꿈 하기 위한 용도
    print(dir_path)
    print(sync)
    np.save(UWB_sync_path, UWB_sync)
    np.save(BIOPAC_sync_path, BIOPAC_sync)




