import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import util
import warnings
warnings.filterwarnings("ignore")


# Raw data extraction from .dat file ======================================
dir_path = "./../Data/2023.01.18/2023.01.18_2_goo_gon"
UWB_data_path = dir_path + "/UWB_cut.npy"
BIOPAC_data_path = dir_path + "/BIOPAC_cut.npy"

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
    Window_UWB_data = np.array(UWB_data[:,get_size * UWB_fs * Window_sliding: get_size * UWB_fs * (Window_sliding + 1)])
    save_dir_path = dir_path + "/" + str(Window_sliding)
    Human = 2
    dynamic_TC = 16

    Human_cnt, Distance, Max_sub_Index = Threshold.dynamic_threshold(Window_sliding)

    for i in range(Human_cnt):
        im = Window_UWB_data[int(Distance[i, 0]):int(Distance[i, 1]) + 1, :]

        L = 0.02

        [u, ux, uy] = util.l0_grad_minimization(im, L)

        Scharr_dx = cv2.Scharr(u.astype("float32"), -1, 1, 0)
        Scharr_dy = cv2.Scharr(u.astype("float32"), -1, 0, 1)
        Scharr_mag = cv2.magnitude(Scharr_dx, Scharr_dy)
        img = cv2.bilateralFilter(Scharr_mag, -1, 5, 10)

        plt.figure(num=3 + i, figsize=(10, 8))
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
        plt.subplot(3, 1, 1)
        plt.subplot(3, 1, 1).set_title("slow index is " + str(Max_sub_Index[i, 0])+ "\n Image smoothing")
        plt.pcolor(img)
        plt.xlabel("Time")
        plt.ylabel("Distance")

        image = save_dir_path + "/" + str(i + 1) + "_person.jpg"
        gray_image = save_dir_path + "/" + str(i + 1) + "_person_gray.jpg"
        print(gray_image)
        plt.subplot(3, 1, 2)
        plt.subplot(3, 1, 2).set_title("Image Gray Scaling")
        plt.imshow(img, cmap="gray")
        plt.imsave(image, img)
        plt.imsave(gray_image, img, cmap="gray")
        plt.xlabel("Time")
        plt.ylabel("Distance")
    plt.show()

