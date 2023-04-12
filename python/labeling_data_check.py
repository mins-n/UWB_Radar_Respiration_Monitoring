import os
import sys

root_dir = "./../Data/"

def get_img_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_gray.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

def get_peak_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_ref.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

def get_UWB_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_UWB.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

data_path_list = get_img_path(root_dir)
ref_path_list = get_peak_path(root_dir)
UWB_path_list = get_UWB_path(root_dir)
data_path_list.sort()
ref_path_list.sort()
UWB_path_list.sort()

for i in range(len(data_path_list)):
    if data_path_list[i][:-18] != UWB_path_list[i][:-17]:
        print(data_path_list[i][:-18])
        print(UWB_path_list[i][:-17])

if len(data_path_list) != len(UWB_path_list):
    print("데이터 크기 불일치")
    sys.exit("오류")