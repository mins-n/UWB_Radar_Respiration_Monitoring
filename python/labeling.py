from PIL import Image
import numpy as np
import os
import re

window_size = 10
uwb_fs = 20
root_dir = "./../Data/"

def get_gray_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_gray.jpg")):
                file_list.append(root + "/" + file_name)
    return file_list

def get_RGB_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
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

def generate_gray_dataset():
    img_path_list = get_gray_path(root_dir)
    for img_path in img_path_list:
        img = []
        load_img = Image.open(img_path).convert("L")
        # print(load_img)
        img_data = np.array(load_img)
        for j in range(0,uwb_fs*61,uwb_fs):
            tmp = img_data[:, j:j+uwb_fs*60]
            img.append(tmp)
        filename = os.path.basename(img_path)
        match = re.search(r'\d+', filename)
        num = int(match.group())
        parent_dir = os.path.dirname(img_path)
        np.save(parent_dir + f"/{num}_person_gray.npy",img)
    return "complete"

def generate_RGB_dataset():
    img_path_list = get_RGB_path(root_dir)
    for img_path in img_path_list:
        img = []
        load_img = Image.open(img_path)
        # print(load_img)
        img_data = np.array(load_img)
        for j in range(0,uwb_fs*61,uwb_fs):
            tmp = img_data[:, j:j+uwb_fs*60]
            img.append(tmp)
        filename = os.path.basename(img_path)
        match = re.search(r'\d+', filename)
        num = int(match.group())
        parent_dir = os.path.dirname(img_path)
        np.save(parent_dir + f"/{num}_person_RGB.npy", img)
    return "complete"

def generate_UWB_dataset():
    UWB_path_list = get_UWB_path(root_dir)
    for UWB_path in UWB_path_list:
        UWB = []
        UWB_data = np.load(UWB_path)
        for j in range(0,uwb_fs*61,uwb_fs):
            tmp = UWB_data[j : j + uwb_fs*60]
            UWB.append(tmp)
        filename = os.path.basename(UWB_path)
        match = re.search(r'\d+', filename)
        num = int(match.group())
        parent_dir = os.path.dirname(UWB_path)
        np.save(parent_dir + f"/{num}_person_UWB_1D.npy", UWB)
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
        for j in range(0,biopac_fs*61,biopac_fs):
            tmp = load_ref
            tmp = tmp[tmp >= j]
            tmp = tmp[tmp <= j+biopac_fs*60]
            RR = len(tmp)
            ref_list.append(RR)
        filename = os.path.basename(ref_path)
        match = re.search(r'\d+', filename)
        num = int(match.group())
        parent_dir = os.path.dirname(ref_path)
        np.save(parent_dir + f"/{num}_person_label.npy", ref_list)
    return "complete"

print(generate_gray_dataset())
print(generate_RGB_dataset())
print(generate_UWB_dataset())
print(generate_RR_ref())