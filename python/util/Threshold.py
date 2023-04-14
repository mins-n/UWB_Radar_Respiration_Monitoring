import os
import numpy as np
import matplotlib.pyplot as plt

class Threshold:
    def __init__(self, dir_path, UWB_data, UWB_Radar_index_start, get_size = 120, UWB_fs = 20, dump_size = 0, show = True):
        self.dir_path = dir_path
        self.UWB_data = UWB_data
        self.UWB_Radar_index_start = UWB_Radar_index_start
        self.get_size = get_size
        self.UWB_fs = UWB_fs
        self.dump_size = dump_size
        self.show = show
        self.fast_to_m = 0.006445 # fast index to meter

    def baseline_threshold(self, Window_UWB_data):
        SD = np.array([])
        for i in range(len(Window_UWB_data)):
            SD = np.append(SD, np.std(Window_UWB_data[i]))  # 거리에 대한 표준편차 배열
        Max = max(SD)  # 가장 큰 표준편차 값
        Index = np.argmax(SD)  # 가장 큰 표준편차 Idx : MAX 위치 값
        Pm = np.mean(SD[Index - 2:Index + 1])  # 가장 높은 표쥰 편차와 -1, +1 idx의 평균값
        Index = Index + self.UWB_Radar_index_start

        d0 = Index * self.fast_to_m  # 가장 높은 편차의 거리 meter
        n = np.mean(SD)  # 편차 배열의 평균

        baselineThreashold = (Pm - n) / (2 * d0 + 1) + n

        return SD, d0, baselineThreashold

    def dynamic_threshold(self, Window_UWB_data, Human = 2, dynamic_TC = 16):
        SD, d0, baselineThreashold = self.baseline_threshold(Window_UWB_data)
        di = np.arange(1, len(self.UWB_data) + 1, 1)
        di = di + self.UWB_Radar_index_start
        di = di * self.fast_to_m

        k = np.array([])
        for i in range(len(di)):
            k = np.append(k, di[i] ** 2 / d0 ** 2)
        k = k ** (-1)
        Dynamic_threshold = np.array(k) * baselineThreashold
        if self.show == True:
            plt.figure(num=1, figsize=(10, 8))
            plt.title("Dynamic Threshold and Standard deviation of raw data")
            plt.plot(Dynamic_threshold)
            plt.plot(SD)
            plt.xlabel("Distance")
            plt.ylabel("standard deviation")
            plt.legend(["Dynamic_threshold", "Standard deviation of raw data"])

        TC_matrix = np.array([])
        Distance = np.zeros((0, 2))

        TC_matrix = SD >= Dynamic_threshold

        for i in range(len(TC_matrix)):
            if TC_matrix[i] == 0 and (i > 2) and (i < len(TC_matrix) - 1):
                if TC_matrix[i - 1] == 1 and TC_matrix[i + 1] == 1:
                    TC_matrix[i] = 1

        TC_cnt = 0
        Human_cnt = 0

        while Human_cnt < Human:
            if self.show == True:
                print("Human_cnt:%d Dynamic_TC:%d" % (Human_cnt, dynamic_TC))
            if dynamic_TC == 1: break
            Human_cnt = 0
            dynamic_TC -= 1
            TC_cnt = 0

            TC_matrix = SD >= Dynamic_threshold

            for i in range(len(TC_matrix)):
                if TC_matrix[i] == 0 and (i > 2) and (i < len(TC_matrix) - 1):
                    if TC_matrix[i - 1] == 1 and TC_matrix[i + 1] == 1:
                        TC_matrix[i] = 1

            for i in range(len(self.UWB_data)):
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

        if Human_cnt > Human:
            Human_cnt = Human

        Max_sub = np.zeros((Human_cnt, 1))
        Max_sub_Index = np.zeros((Human_cnt, 1))

        for i in range(Human_cnt):
            Max_sub[i, 0] = max(SD[int(Distance[i, 0]) - 1:int(Distance[i, 1])])
            Max_sub_Index[i, 0] = np.argmax(SD[int(Distance[i, 0]) - 1:int(Distance[i, 1])])

            Max_sub_Index[i, 0] += Distance[i, 0]

            if len(self.UWB_data) < Max_sub_Index[i, 0] + 15:
                Distance[i, 0] = Max_sub_Index[i, 0] - 15
                Distance[i, 1] = len(self.UWB_data[1])
            elif Max_sub_Index[i, 0] - 15 < 1:
                Distance[i, 0] = 1
                Distance[i, 1] = Max_sub_Index[i, 0] + 15
            else:
                Distance[i, 0] = Max_sub_Index[i, 0] - 15
                Distance[i, 1] = Max_sub_Index[i, 0] + 15

        Data = TC_matrix.reshape(TC_matrix.size, 1) * self.UWB_data

        if Human_cnt > Human:
            Human_cnt = Human  # 2명보다 많으면 2명으로 고정

        return Human_cnt, Distance, Max_sub_Index