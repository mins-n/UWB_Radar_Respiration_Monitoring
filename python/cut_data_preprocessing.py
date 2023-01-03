import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import warnings
warnings.filterwarnings("ignore")

def gradients(I):
    N = len(I)
    M = len(I[0])

    Ix = np.zeros((N, M))
    Iy = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            if i < N:
                Ix[i - 1, j - 1] = I[i, j - 1] - I[i - 1, j - 1]
            else:
                Ix[N - 1, j - 1] = I[0, j - 1] - I[N - 1, j - 1]

            if j < M:
                Iy[i - 1, j - 1] = I[i - 1, j] - I[i - 1, j - 1]
            else:
                Iy[i - 1, M - 1] = I[i - 1, 0] - I[i - 1, M - 1]

    return Ix, I
def l0_grad_minimization(y, L):
    N = len(y)
    M = len(y[0])

    u = 1.0 * y

    h = np.zeros((N, M))
    v = np.zeros((N, M))

    p = np.zeros((N, M))
    q = np.zeros((N, M))

    fy = np.zeros((N, M))
    fy = np.fft.fft2(y)

    dx = np.zeros((N, M))
    dy = np.zeros((N, M))
    dx[1, 1] = -1.0
    dx[N - 1, 1] = 1.0
    dy[1, 1] = -1.0
    dy[1, M - 1] = 1.0

    fdxt = np.tile(np.conj(np.fft.fft2(dx)), [1, 1])
    fdyt = np.tile(np.conj(np.fft.fft2(dy)), [1, 1])

    adxy = abs(fdxt) ** 2 + abs(fdyt) ** 2

    beta = 0.5 / L
    for t in range(50):
        if beta <= 1e-2:
            break
        np.disp(t)
        np.disp(beta)

        [ux, uy] = gradients(u)

        ls = abs(ux) ** 2 + abs(uy) ** 2 >= beta * L
        h = ls * ux
        v = ls * uy

        fh = np.zeros((N, M))
        fv = np.zeros((N, M))
        fh = np.fft.fft2(h)
        fv = np.fft.fft2(v)

        fu = (beta * fy + (fdxt * fh + fdyt * fv)) / (beta + adxy)
        u = np.real(np.fft.ifft2(fu))

        beta = 0.65 * beta

    return u, h, v

# Raw data extraction from .dat file ======================================
dir_path = "./../Data/2023.01.02/2023.01.02_3_sun_chan"
UWB_data_path = dir_path + "/UWB_cut.npy"
BIOPAC_data_path = dir_path + "/BIOPAC_cut.npy"
UWB_data = np.load(UWB_data_path)
BIOPAC_data = np.load(BIOPAC_data_path, allow_pickle=True)
BIOPAC_data_1 = BIOPAC_data[0]
BIOPAC_fs_1 = BIOPAC_data[1]
BIOPAC_data_2 = BIOPAC_data[2]
BIOPAC_fs_2 = BIOPAC_data[3]

Window_UWB_data = []
window_BIOPAC_data_1 = []
window_BIOPAC_data_2 = []
UWB_fs = 20
get_size = 120 # second
dump_size = 30 # second
fast_to_m = 0.006445  # fast index to meter
UWB_Radar_index_start = 0.5  # UWB Radar Range 0.5 ~ 2.5m
UWB_Radar_index_start = math.floor(UWB_Radar_index_start / fast_to_m)

## Baseline Threshold ==========================================================================
for Window_sliding in range(5):
    save_dir_path = dir_path + "/" + str(Window_sliding)
    BIOPAC_data_save_path = save_dir_path + "/BIOPAC_data.npy"
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    Window_UWB_data = np.array(UWB_data[:, Window_sliding * get_size * UWB_fs + Window_sliding * dump_size * UWB_fs: (Window_sliding + 1) * get_size * UWB_fs + Window_sliding * dump_size * UWB_fs])
    window_BIOPAC_data_1 = np.array(BIOPAC_data_1[Window_sliding * get_size * BIOPAC_fs_1: (Window_sliding + 1) * get_size * BIOPAC_fs_1])
    window_BIOPAC_data_2 = np.array(BIOPAC_data_2[Window_sliding * get_size * BIOPAC_fs_2: (Window_sliding + 1) * get_size * BIOPAC_fs_2])
    BIOPAC_data = []
    BIOPAC_data.append(window_BIOPAC_data_1)
    BIOPAC_data.append(BIOPAC_fs_1)
    BIOPAC_data.append(window_BIOPAC_data_2)
    BIOPAC_data.append(BIOPAC_fs_2)
    np.save(BIOPAC_data_save_path, BIOPAC_data)

    SD = np.array([])
    for i in range(len(Window_UWB_data)):
        SD = np.append(SD, np.std(Window_UWB_data[i]))  # 거리에 대한 표준편차 배열
    Max = max(SD)  # 가장 큰 표준편차 값
    Index = np.argmax(SD)  # 가장 큰 표준편차 Idx : MAX 위치 값
    Pm = np.mean(SD[Index - 2:Index + 1])  # 가장 높은 표쥰 편차와 -1, +1 idx의 평균값
    Index = Index + UWB_Radar_index_start

    d0 = Index * fast_to_m  # 가장 높은 편차의 거리 meter
    n = np.mean(SD)  # 편차 배열의 평균

    baselineThreashold = (Pm - n) / (2 * d0 + 1) + n

    #Dynamic Threshold ===========================================================================
    di = np.arange(1, len(UWB_data) + 1, 1)
    di = di + UWB_Radar_index_start
    di = di * fast_to_m

    k = np.array([])
    for i in range(len(di)):
        k = np.append(k, di[i] ** 2 / d0 ** 2)
    k = k ** (-1)
    Dynamic_threshold = np.array(k) * baselineThreashold

    plt.figure(num=1,figsize=(10, 8))
    plt.title("Dynamic Threshold and Standard deviation of raw data")
    plt.plot(Dynamic_threshold)
    plt.plot(SD)
    plt.xlabel('Distance')
    plt.ylabel('standard deviation')
    plt.legend(["Dynamic_threshold","Standard deviation of raw data"])

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
    for i in range(len(UWB_data)):
        if TC_matrix[i]:
            TC_cnt += 1
        else:
            if TC_cnt < 15:
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
        if TC_cnt < 15:
            TC_matrix[i-TC_cnt :] = 0
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

        if len(UWB_data) < Max_sub_Index[i, 0] + 15:
            Distance[i, 0] = Max_sub_Index[i, 0] - 15
            Distance[i, 1] = len(UWB_data[1])
        elif Max_sub_Index[i, 0] - 15 < 1:
            Distance[i, 0] = 1
            Distance[i, 1] = Max_sub_Index[i, 0] + 15
        else:
            Distance[i, 0] = Max_sub_Index[i, 0] - 15
            Distance[i, 1] = Max_sub_Index[i, 0] + 15

    Data = TC_matrix.reshape(TC_matrix.size,1) * UWB_data

    if Human_cnt > 2:
        Human_cnt = 2  # 2명보다 많으면 2명으로 고정(임시) -최광진
    for i in range(Human_cnt):
        im = Window_UWB_data[int(Distance[i, 0]):int(Distance[i, 1]) + 1, :]

        L = 0.02
        [u, ux, uy] = l0_grad_minimization(im, L)

        Scharr_dx = cv2.Scharr(u.astype('float32'), -1, 1, 0)
        Scharr_dy = cv2.Scharr(u.astype('float32'), -1, 0, 1)
        Scharr_mag = cv2.magnitude(Scharr_dx, Scharr_dy)
        img = cv2.bilateralFilter(Scharr_mag, -1, 5, 10)

        plt.figure(num=3 + i, figsize=(10, 8))
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
        plt.subplot(3, 1, 1)
        plt.subplot(3, 1, 1).set_title("slow index is " + str(Max_sub_Index[i, 0])+ "\n Image smoothing")
        plt.pcolor(img)
        plt.xlabel('Time')
        plt.ylabel('Distance')

        image = save_dir_path + "/" + str(i + 1) + "_person.jpg"
        gray_image = save_dir_path + "/" + str(i + 1) + "_person_gary.jpg"
        print(gray_image)
        plt.subplot(3, 1, 2)
        plt.subplot(3, 1, 2).set_title("Image Gray Scaling")
        plt.imshow(img, cmap='gray')
        plt.imsave(image, img)
        plt.imsave(gray_image, img, cmap='gray')
        plt.xlabel('Time')
        plt.ylabel('Distance')
    plt.show()

