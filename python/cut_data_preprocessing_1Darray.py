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
# Raw data extraction from .dat file ======================================
for dir_path in tqdm(file_list):
    UWB_data_path = dir_path + "/UWB_cut.npy"
    if not os.path.exists(UWB_data_path):
        UWB_data_path = dir_path + "/UWB_sync.npy"

    BIOPAC_data_path = dir_path + "/BIOPAC_cut.npy"
    if not os.path.exists(BIOPAC_data_path):
        BIOPAC_data_path = dir_path + "/BIOPAC_sync.npy"

    UWB_data = np.load(UWB_data_path)
    BIOPAC_data = np.load(BIOPAC_data_path, allow_pickle=True)


    Window_UWB_data = []
    window_BIOPAC_data_1 = []
    window_BIOPAC_data_2 = []
    UWB_fs = 20
    get_size = 120 # second
    dump_size = 30 # second
    fast_to_m = 0.006445  # fast index to meter
    UWB_Radar_index_start = 0.5  # UWB Radar Range 0.5 ~ 2.5m
    UWB_Radar_index_start = math.floor(UWB_Radar_index_start / fast_to_m)

    Threshold = util.Threshold(dir_path, UWB_data, BIOPAC_data, UWB_Radar_index_start, get_size, UWB_fs)

    for Window_sliding in range(int(len(UWB_data[0]) / (get_size * UWB_fs))):
        save_dir_path = dir_path + "/" + str(Window_sliding)
        if not os.path.exists(save_dir_path):
            continue

        Window_UWB_data = np.array(UWB_data[:,get_size * UWB_fs * Window_sliding: get_size * UWB_fs * (Window_sliding + 1)])
        Human = 2
        dynamic_TC = 16

        Human_cnt, Distance, Max_sub_Index = Threshold.dynamic_threshold(Window_sliding)

        for i in range(Human_cnt):
            save_UWB_path = save_dir_path + "/" + str(i + 1) + "_person_UWB.npy"
            save_UWB = Window_UWB_data[int(Max_sub_Index[i,0]), :]
            np.save(save_UWB_path,save_UWB)
