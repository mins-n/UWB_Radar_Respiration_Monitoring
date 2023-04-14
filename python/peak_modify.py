import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import util
import os

base_path = "./../Data/2023.01.04/2023.01.04_3_gon_gu"
base_path = base_path + "/"


BIOPAC_folder_path = base_path + str(4) + "/"
###
in_data = [

[ 1491, 3387,  5656, 6637, 7616, 8549, 9533,
10270, 12172, 13694,  16442,  18846,
21291,23508, 26069, 28448, 30821, 33097, 35151, 37089,
39157, 41684, 44149, 46501, 49270, 51927, 54368, 56608,
59037],

[1216, 3313, 4515, 6933,  9335, 10515, 11299, 11919,17380,47610,58740,
12649, 14208, 15824, 18808, 20889, 23319, 25855, 28314, 30391,
32673, 34860, 37205,  39870, 42300, 44282, 45994,
48945, 50567,52874, 55021, 56902]

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

fig, axs = plt.subplots(4, 1, figsize=(18, 10))

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