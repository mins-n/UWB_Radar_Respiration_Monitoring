import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import util
from tqdm import tqdm
import warnings
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
error_dir = root_dir + "error.txt"
file_list = get_data_path(root_dir)
error_list = []
for dir_path in tqdm(file_list[22:]):
    UWB_data_path = dir_path + "/UWB_sync.npy"
    BIOPAC_data_path = dir_path + "/BIOPAC_sync.npy"
    ori_UWB_data = np.load(UWB_data_path)
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
    dump_size = 30  # second
    if len(ori_UWB_data[0]) > 15500:
        UWB_data = ori_UWB_data[:, 35 * UWB_fs:]
    elif len(ori_UWB_data[0]) > 14900:
        UWB_data = ori_UWB_data[:,30 * UWB_fs:]
    else:
        if len(ori_UWB_data[0]) > 12600:
            UWB_data = ori_UWB_data[:,400:]
        elif len(ori_UWB_data[0]) > 12500:
            UWB_data = ori_UWB_data[:, 300:]
        elif len(ori_UWB_data[0]) > 12400:
            UWB_data = ori_UWB_data[:, 200:]
        elif len(ori_UWB_data[0]) > 12300:
            UWB_data = ori_UWB_data[:, 100:]
        else:
            UWB_data = ori_UWB_data
        dump_size = 0  # second

    fast_to_m = 0.006445  # fast index to meter
    UWB_Radar_start = 0.5  # UWB Radar Range 0.5 ~ 2.5m
    UWB_Radar_index_start = math.floor(UWB_Radar_start / fast_to_m)
    show = False # True: plt 출력, False: plt 출력 안함

    info = dir_path + "/info.txt"
    with open(info, "w", encoding="utf8") as f:
        f.write(f"UWB 데이터 길이: {round(len(ori_UWB_data[0])/(UWB_fs*60),2)}분\n")
        f.write(f"UWB fs: {UWB_fs}\n")
        f.write(f"BIOPAC 데이터 길이: {round(len(BIOPAC_data_1) / (BIOPAC_fs_1 * 60), 2)}분\n")
        f.write(f"BIOPAC fs: {BIOPAC_fs_1}\n\n")

    Threshold = util.Threshold(dir_path, UWB_data, UWB_Radar_index_start, get_size, UWB_fs, dump_size, show)

    for Window_sliding in range(5):
        save_dir_path = dir_path + "/" + str(Window_sliding)
        BIOPAC_data_save_path = save_dir_path + "/BIOPAC_data.npy"

        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)

        Human = 2
        dynamic_TC = 16

        Window_UWB_data = np.array(
            UWB_data[:,(get_size + dump_size) * UWB_fs * Window_sliding:
            get_size * UWB_fs * (Window_sliding + 1) + Window_sliding * dump_size * UWB_fs])
        window_BIOPAC_data_1 = np.array(
            BIOPAC_data_1[(get_size + dump_size) * BIOPAC_fs_1 * Window_sliding:
            ( Window_sliding + 1) * get_size * BIOPAC_fs_1 + Window_sliding * dump_size * BIOPAC_fs_1])
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

        if Human_cnt == 0:
            error_list.append(save_dir_path)
            continue

        if Human_cnt < Human:
            error_list.append(save_dir_path)

        for i in range(Human_cnt):
            save_UWB_path = save_dir_path + "/" + str(i + 1) + "_person_UWB.npy"
            save_UWB = Window_UWB_data[int(Max_sub_Index[i, 0]), :]
            np.save(save_UWB_path, save_UWB)

            im = Window_UWB_data[int(Distance[i, 0]):int(Distance[i, 1]) + 1, :]

            L = 0.02
            [u, ux, uy] = util.l0_grad_minimization(im, L)

            # check_path = save_dir_path + "/" + str(i + 1) + "_person.npy"
            # if not os.path.exists(check_path):
            #     continue

            Scharr_dx = cv2.Scharr(u.astype("float32"), -1, 1, 0)
            Scharr_dy = cv2.Scharr(u.astype("float32"), -1, 0, 1)
            Scharr_mag = cv2.magnitude(Scharr_dx, Scharr_dy)
            img = cv2.bilateralFilter(Scharr_mag, -1, 5, 10)

            if show == True:
                plt.figure(num=3 + i, figsize=(10, 8))
                plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
                plt.subplot(3, 1, 1)
                plt.subplot(3, 1, 1).set_title("slow index is " + str(Max_sub_Index[i, 0])+ "\n Image smoothing")
                plt.pcolor(img)
                plt.xlabel("Time")
                plt.ylabel("Distance")

                plt.subplot(3, 1, 2)
                plt.subplot(3, 1, 2).set_title("Image Gray Scaling")
                plt.imshow(img, cmap="gray")
                plt.xlabel("Time")
                plt.ylabel("Distance")

            image = save_dir_path + "/" + str(i + 1) + "_person.jpg"
            gray_image = save_dir_path + "/" + str(i + 1) + "_person_gray.jpg"
            plt.imsave(image, img)
            plt.imsave(gray_image, img, cmap="gray")

        info = save_dir_path + "/info.txt"

        with open(info, "w", encoding="utf8") as f:
            f.write(f"데이터 길이: {get_size}초\n")
            f.write(f"UWB fs: {UWB_fs}\n")
            f.write(f"BIOPAC fs: {BIOPAC_fs_1}\n\n")
            f.write(f"{UWB_Radar_start}m 부터 측정\n")
            f.write(f"{UWB_Radar_index_start}index 부터 측정\n\n")
            f.write(f"인식된 사람수: {Human_cnt}명\n")
            f.write(f"첫번째 사람의 fast index: {Max_sub_Index[0, 0]}\n")
            f.write(f"첫번째 사람의 meter: {round((Max_sub_Index[0, 0] + UWB_Radar_index_start) * fast_to_m, 2)}m\n\n")
            if Human_cnt == 1:
                continue
            f.write(f"두번째 사람의 fast index: {Max_sub_Index[1, 0]}\n")
            f.write(f"두번째 사람의 meter: {round((Max_sub_Index[1, 0] + UWB_Radar_index_start) * fast_to_m,2)}m\n")

        if show == True:
            plt.show()

with open(error_dir, "w", encoding="utf8") as f:
    f.write("아래 경로 사람수 2 이하 오류")
    for err in error_list:
        f.write(err + "\n")

