import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import util
import os

base_path = "./../Data/2023.01.18/2023.01.18_2_goo_gon"
base_path = base_path + "/"


BIOPAC_folder_path = base_path + str(4) + "/"
###
in_data = [
[1459, 4835, 8082, 15262, 18460, 21422, 24509, 27441,11730,
30432, 33297, 36409, 39460, 42543, 45433, 48610, 52096, 55374,
58896],

[1219, 3803, 6900, 9595, 12103,15650, 18203, 21314,
24108, 27428, 32045, 34875,  40590, 45016,48220,
51132, 53939,
56524, 59308]

    ]
###

check_1 = BIOPAC_folder_path + "1_person_UWB.npy"
check_2 = BIOPAC_folder_path + "2_person_UWB.npy"

BIOPAC_path = BIOPAC_folder_path + "BIOPAC_data.npy"

BIOPAC_rpeak_i = []

data = np.load(BIOPAC_path, allow_pickle=True)

data1 = data[0]  # 1번째 사람 데이터
data1_fs = data[1]  # 바이오팩 fs

data2 = data[2]
data2_fs = data[3]

# Set the sample rate and minimum interval
MI = 1  # Minimum interval in seconds

data1_rpeak_i, data1_env_peak_i = util.Peak_Detection(data1, data1_fs, MI)  # peak detection
data1_rpeak_i = data1_rpeak_i.astype(int)
BIOPAC_rpeak_i.append(data1_rpeak_i)

data2_rpeak_i, data2_env_peak_i = util.Peak_Detection(data2, data2_fs, MI)
data2_rpeak_i = data2_rpeak_i.astype(int)
BIOPAC_rpeak_i.append(data2_rpeak_i)

# Print the detected peaks
print("\n[", end="")
print_len = 9
for i in range(int(len(data1_rpeak_i[:-1]) / print_len) + 1):
    if (print_len * (i + 1) > len(data1_rpeak_i[:-1])):
        print(*data1_rpeak_i[print_len * i:-1].tolist(), sep=', ', end="],\n\n")
    else:
        print(*data1_rpeak_i[print_len * i:print_len * (i + 1)].tolist(), sep=', ', end=',\n')
print("[", end="")
for i in range(int(len(data2_rpeak_i[:-1]) / print_len) + 1):
    if (print_len * (i + 1) > len(data2_rpeak_i[:-1])):
        print(*data2_rpeak_i[print_len * i:-1].tolist(), sep=', ', end="]\n\n")
    else:
        print(*data2_rpeak_i[print_len * i:print_len * (i + 1)].tolist(), sep=', ', end=',\n')


###
data1_rpeak_i = in_data[0]
data2_rpeak_i = in_data[1]
###

fig, axs = plt.subplots(4, 1, figsize=(18, 9))

if os.path.exists(check_1):

    axs[0].set_xlim(0, data1_fs * 60)
    axs[0].plot(data1)
    axs[0].plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')
    axs[0].set_title("BIOPAC Peak Detection Data1")

    axs[1].set_xlim(data1_fs * 60, data1_fs * 120)
    axs[1].plot(data1)
    axs[1].plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')


if os.path.exists(check_2):

    axs[2].set_xlim(0, data2_fs * 60)
    axs[2].plot(data2)
    axs[2].plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')
    axs[2].set_title("BIOPAC Peak Detection Data2")

    axs[3].set_xlim(data2_fs * 60, data2_fs * 120)
    axs[3].plot(data2)
    axs[3].plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')

# set the x-axis tick interval to 1000 units
for ax in axs:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(data2_fs*2))

#저장

###
if os.path.exists(check_1):
    data1_rpeak_i = np.sort(np.array(in_data[0]))
    np.save(BIOPAC_folder_path + 'data1_rpeak_i', data1_rpeak_i)

if os.path.exists(check_2):
    data2_rpeak_i = np.sort(np.array(in_data[1]))
    np.save(BIOPAC_folder_path +'data2_rpeak_i', data2_rpeak_i)
###


plt.tight_layout()
plt.show()