from PIL import Image
import numpy as np
import os

window_size = 10
uwb_fs = 20
root_dir = "./../Data/"

def get_img_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_gray.jpg")):
                file_list.append(root + "/" + file_name)
            if (file_name.endswith("_person.jpg")):
                file_list.append(root + "/" + file_name)
    return file_list

def get_UWB_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_UWB.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

def get_peak_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_rpeak_i.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

def generate_img_dataset():
    img_path_list = get_img_path(root_dir)
    for img_path in img_path_list:
        img = []
        load_img = Image.open(img_path).convert("L")
        # print(load_img)
        img_data = np.array(load_img)
        for j in range(110):
            tmp = img_data[:, j * uwb_fs:(10 + j) * uwb_fs]
            img.append(tmp)
        np.save(img_path[:-4] + ".npy",img)
    return "complete"

def generate_UWB_dataset():
    UWB_path_list = get_UWB_path(root_dir)
    for UWB_path in UWB_path_list:
        UWB = []
        UWB_data = np.load(UWB_path)
        for j in range(110):
            tmp = UWB_data[j * uwb_fs:(10 + j) * uwb_fs]
            UWB.append(tmp)
        np.save(UWB_path[:-4] + "_input.npy", UWB)
    return "complete"
def generate_RR_ref():
    ref_path_list = get_peak_path(root_dir)
    for ref_path in ref_path_list:
        ref_list = []
        load_ref = np.load(ref_path)
        load_ref = np.sort(load_ref)
        load_ref = np.unique(load_ref)
        dir__ = os.path.dirname(ref_path)
        dir__ = dir__ + "\\BIOPAC_data.npy"
        biopac_fs = np.load(dir__,allow_pickle=True)[1]
        for j in range(110):
            tmp = load_ref
            tmp = tmp[tmp > j * biopac_fs]
            tmp = tmp[tmp < (10 + j) * biopac_fs]
            if len(tmp) <= 1:
                k = 1
                while len(tmp) < 2:
                    tmp = load_ref
                    tmp = tmp[tmp > (j-k) * biopac_fs]
                    tmp = tmp[tmp < (10 + j + k) * biopac_fs]
                    k = k + 1
            diffs = np.diff(tmp)
            tmp_diffs = np.mean(diffs)
            RR = round(60/(tmp_diffs/biopac_fs),2)
            ref_list.append(RR)
        # print(ref_list)
        np.save(ref_path[:-12] + "_ref.npy",ref_list)
    return "complete"
def generate_dot_ref():
    ref_path_list = get_peak_path(root_dir)
    for ref_path in ref_path_list:
        ref_list = []
        load_ref = np.load(ref_path)
        load_ref = np.sort(load_ref)
        load_ref = np.unique(load_ref)
        dir__ = os.path.dirname(ref_path)
        dir__ = dir__ + "\\BIOPAC_data.npy"
        biopac_fs = np.load(dir__,allow_pickle=True)[1]
        for j in range(110):
            tmp = load_ref
            tmp = tmp[tmp > j * biopac_fs]
            tmp = tmp[tmp < (10 + j) * biopac_fs]
            ref_list.append(len(tmp))
        np.save(ref_path[:-12] + "_ref.npy",ref_list)
    return "complete"

print(generate_img_dataset())
print(generate_UWB_dataset())
print(generate_RR_ref())