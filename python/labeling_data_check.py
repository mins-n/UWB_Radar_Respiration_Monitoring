import os
import sys
import numpy as np
root_dir = "./../Data/"

def get_gray_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_gray.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

def get_RGB_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_RGB.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

def get_peak_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_label.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

def get_UWB_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_UWB_1D.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

data_path_list = get_RGB_path(root_dir)
ref_path_list = get_peak_path(root_dir)
UWB_path_list = get_UWB_path(root_dir)
data_path_list.sort()
ref_path_list.sort()
UWB_path_list.sort()

for i in range(len(data_path_list)):
    if os.path.dirname(data_path_list[i]) != os.path.dirname(UWB_path_list[i]) or os.path.dirname(UWB_path_list[i]) != os.path.dirname(ref_path_list[i]):
        print(data_path_list[i])
        print(UWB_path_list[i])
        print(ref_path_list[i])
        sys.exit()

# for i in range(len(data_path_list)):
#     if os.path.exists(data_path_list[i]):
#         os.remove(data_path_list[i])
#     if os.path.exists(UWB_path_list[i]):
#         os.remove(UWB_path_list[i])
#     if os.path.exists(ref_path_list[i]):
#         os.remove(ref_path_list[i])

# for i in range(len(data_path_list)):
#     print(os.path.dirname(data_path_list[i]))
#     print(f"image: {np.load(data_path_list[i]).shape}")
#     print(f"UWB: {np.load(UWB_path_list[i]).shape}")
#     print(f"label: {np.load(ref_path_list[i]).shape}\n")

if len(data_path_list) != len(UWB_path_list):
    print("데이터 크기 불일치")
    sys.exit("오류")