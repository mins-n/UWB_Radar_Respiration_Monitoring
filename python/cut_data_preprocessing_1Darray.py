import math
import numpy as np
import util
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

def get_data_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("UWB_sync.npy")):
                root = root.replace("\\", "/")
                file_list.append(root)
    return file_list
root_dir = "./../Data/"
file_list = get_data_path(root_dir)
for dir_path in tqdm(file_list):
    UWB_data_path = dir_path + "/UWB_cut.npy"
    if not os.path.exists(UWB_data_path):
        UWB_data_path = dir_path + "/UWB_sync.npy"

    BIOPAC_data_path = dir_path + "/BIOPAC_cut.npy"
    if not os.path.exists(BIOPAC_data_path):
        BIOPAC_data_path = dir_path + "/BIOPAC_sync.npy"

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

    if len(UWB_data[0]) > 13900:
        dump_size = 30 # second
    else:
        dump_size = 0  # second
    fast_to_m = 0.006445  # fast index to meter
    UWB_Radar_index_start = 0.5  # UWB Radar Range 0.5 ~ 2.5m
    UWB_Radar_index_start = math.floor(UWB_Radar_index_start / fast_to_m)
    show = False  # True: plt 출력, False: plt 출력 안함

    Threshold = util.Threshold(dir_path, UWB_data, UWB_Radar_index_start, get_size, UWB_fs, dump_size, show)

    for Window_sliding in range(5):
        save_dir_path = dir_path + "/" + str(Window_sliding)
        BIOPAC_data_save_path = save_dir_path + "/BIOPAC_data.npy"
        # if save_dir_path == "./../Data/2023.01.10/2023.01.10_6_soo_jin/4":
        #     print(11)
        if not os.path.exists(save_dir_path):
            continue

        Window_UWB_data = np.array(
            UWB_data[:, (get_size + dump_size) * UWB_fs * Window_sliding:
                        get_size * UWB_fs * (Window_sliding + 1) + Window_sliding * dump_size * UWB_fs])
        window_BIOPAC_data_1 = np.array(
            BIOPAC_data_1[(get_size + dump_size) * BIOPAC_fs_1 * Window_sliding:
                          (Window_sliding + 1) * get_size * BIOPAC_fs_1 + Window_sliding * dump_size * BIOPAC_fs_1])
        window_BIOPAC_data_2 = np.array(
            BIOPAC_data_2[(get_size + dump_size) * BIOPAC_fs_2 * Window_sliding:
                          (Window_sliding + 1) * get_size * BIOPAC_fs_2 + Window_sliding * dump_size * BIOPAC_fs_2])

        BIOPAC_data = []
        BIOPAC_data.append(window_BIOPAC_data_1)
        BIOPAC_data.append(BIOPAC_fs_1)
        BIOPAC_data.append(window_BIOPAC_data_2)
        BIOPAC_data.append(BIOPAC_fs_2)
        np.save(BIOPAC_data_save_path, BIOPAC_data)

        Human_cnt, Distance, Max_sub_Index = Threshold.dynamic_threshold(Window_UWB_data)

        for i in range(Human_cnt):
            check_path = save_dir_path + "/" + str(i + 1) + "_person.npy"
            if not os.path.exists(check_path):
                continue
            save_UWB_path = save_dir_path + "/" + str(i + 1) + "_person_UWB.npy"
            save_UWB = Window_UWB_data[int(Max_sub_Index[i,0]), :]
            np.save(save_UWB_path,save_UWB)
