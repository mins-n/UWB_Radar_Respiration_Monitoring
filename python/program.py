import math
import numpy as np
import cv2
import os
from PIL import Image
import warnings
import neurokit2 as nk
import matplotlib.pyplot as plt
import tensorflow as tf

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

model = tf.keras.models.load_model("./respiration_rate_predict")

# Raw data extraction from .dat file ======================================
dir_path = "./../Data/2022.12.26/2022.12.26_3_soo_jin"
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

# fastindex당 0.6445cm slowindex 1초당 20index
fast_to_m = 0.006445  # fast index to meter
UWB_Radar_index_start = 0.5  # UWB Radar Range 0.5 ~ 2.5m
UWB_Radar_index_start = math.floor(UWB_Radar_index_start / fast_to_m)

Windowsize = 1200 # 60sec

## Baseline Threshold ==========================================================================
for Window_sliding in range(int(len(rawdata[0]) / Windowsize) + 1):
    if Window_sliding == int(len(rawdata[0]) / Windowsize):
        Window_rawdata = np.array(rawdata[:, Windowsize * Window_sliding:])
    else:
        Window_rawdata = np.array(rawdata[:, Windowsize * Window_sliding:Windowsize * (Window_sliding + 1)])

SD = np.array([])
for i in range(len(rawdata)):
    SD = np.append(SD, np.std(rawdata[i]))  # 거리에 대한 표준편차 배열
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

while Human_cnt < Human:
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

if Human_cnt > 2:
    Human_cnt = 2  # 2명보다 많으면 2명으로 고정

tmp_gray_path = "./tmp_gray.jpg"
uwb_fs = 20
rsp = []
average_rsp = []
for n in range(Human_cnt):
    im = rawdata[int(Distance[n, 0]):int(Distance[n, 1]) + 1, :]

    L = 0.02
    [u, ux, uy] = l0_grad_minimization(im, L)

    Scharr_dx = cv2.Scharr(u.astype('float32'), -1, 1, 0)
    Scharr_dy = cv2.Scharr(u.astype('float32'), -1, 0, 1)
    Scharr_mag = cv2.magnitude(Scharr_dx, Scharr_dy)
    img = cv2.bilateralFilter(Scharr_mag, -1, 5, 10)
    plt.imsave(tmp_gray_path, img, cmap='gray')
    load_img = np.array(Image.open(tmp_gray_path).convert("L"))
    tmp = []
    gray_image = []
    for j in range(0,math.trunc(len(rawdata[0])),200): # 10 초 fs 20, 10 * 20 == 200
        tmp = load_img[:, j :(200 + j)]
        gray_image.append(tmp)
    res_predict = []
    data_list = []

    for i in range(len(gray_image)):
        if not gray_image[i].any():
            continue  # Skip this iteration if the image is empty
        resized_img = cv2.resize(gray_image[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        data_list.append(resized_img)
    data_list = np.array(data_list)
    data_list = data_list.reshape(len(data_list), 28, 28, 1)
    data_list = data_list / 255.0


    res_predict = model.predict(data_list)

    tmp_rsp = []
    len_res = 0
    average = 0
    for res in res_predict:
        for i,k in enumerate(res):
            try:
                tmp_rsp = tmp_rsp + nk.rsp_simulate(duration=10, sampling_rate=500, respiratory_rate=k).tolist()
                average = average + k
                len_res = len_res + 1
            except:
                print(f"{n}번쨰 사람이 {i*10}초에서 {(i+1)*10} 사이에 예측된 호흡이 {k}로 불안정하다.")
                tmp_rsp = tmp_rsp + [0] * 5000
    average_rsp.append(average/len_res)
    rsp.append(tmp_rsp)

print("\n 총 " + str(round(len(res_predict)*10/60)) + "분 데이터")
print(str(round(Max_sub_Index[0, 0]*0.6445/100 + 0.5,2)) + f"m에 위치한 사람의 평균 호흡수 {average_rsp[0]}")
print(str(round(Max_sub_Index[1, 0]*0.6445/100 + 0.5,2)) + f"m에 위치한 사람의 평균 호흡수 {average_rsp[1]}")

plt.figure(num=1, figsize=(20, 14))
plt.title("Plot the respiratory rate of CNN results")
plt.plot(rsp[0])
plt.plot(rsp[1])
plt.xlabel('Time, 500fs')
plt.ylabel('Amplitude')
plt.legend([f"{round(Max_sub_Index[0, 0]*0.6445/100 + 0.5,2)}m Person",f"{round(Max_sub_Index[1, 0]*0.6445/100 + 0.5,2)}m Person"])
plt.show()
