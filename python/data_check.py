import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_path = "./../Data/2023.01.04/2023.01.04_1_gu_gon"
base_path = base_path + "/"

for i in range(5):
    dir_path = base_path + str(i)
    if not os.path.exists(dir_path):
        print(str(i) + "번 폴더 없음")
        continue
    dir_path = dir_path + "/"
    BIOPAC_data = np.load(dir_path + "BIOPAC_data.npy", allow_pickle=True)
    BIOPAC_data_1 = BIOPAC_data[0]
    BIOPAC_fs_1 = BIOPAC_data[1]

    BIOPAC_data_2 = BIOPAC_data[2]
    BIOPAC_fs_2 = BIOPAC_data[3]

    check_1 = dir_path + "1_person_UWB.npy"
    check_2 = dir_path + "2_person_UWB.npy"

    plt.figure(num=i)
    fig, axs = plt.subplots(8, 1, figsize=(10, 16))

    if not os.path.exists(check_1):
        print(str(i) + "번 폴더 1번 사람 측정 안됨")
    else:
        img1 = mpimg.imread(dir_path + "1_person.jpg")
        gray1 = mpimg.imread(dir_path + "1_person_gray.jpg")
        uwb1 = np.load(dir_path + "1_person_UWB.npy")

        axs[0].imshow(img1)
        axs[0].set_title("1_person.jpg")

        axs[1].imshow(gray1, cmap="gray")
        axs[1].set_title("1_person_gray.jpg")

        axs[2].plot(uwb1)
        axs[2].set_title("1_person_UWB.npy")
        axs[2].set_xlim(0, len(uwb1))  # set x-axis limits to match image subplots

        axs[3].plot(BIOPAC_data_1)
        axs[3].set_title("1_person_BIOPAC")
        axs[3].set_xlim(0, len(BIOPAC_data_1))  # set x-axis limits to match image subplots

    if not os.path.exists(check_2):
        print(str(i) + "번 폴더 2번 사람 측정 안됨")
    else:
        img2 = mpimg.imread(dir_path + "2_person.jpg")
        gray2 = mpimg.imread(dir_path + "2_person_gray.jpg")
        uwb2 = np.load(dir_path + "2_person_UWB.npy")

        axs[4].imshow(img2)
        axs[4].set_title("2_person.jpg")

        axs[5].imshow(gray2, cmap="gray")
        axs[5].set_title("2_person_gray.jpg")

        axs[6].plot(uwb2)
        axs[6].set_title("2_person_UWB.npy")
        axs[6].set_xlim(0, len(uwb2)) # set x-axis limits to match image subplots

        axs[7].plot(BIOPAC_data_2)
        axs[7].set_title("2_person_BIOPAC")
        axs[7].set_xlim(0, len(BIOPAC_data_2)) # set x-axis limits to match image subplots
    plt.tight_layout()

plt.show()
