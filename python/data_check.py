import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


dir_path = "./../Data/2023.01.18/2023.01.18_2_goo_gon/0/"

img1 = mpimg.imread(dir_path + "1_person.jpg")
gray1 = mpimg.imread(dir_path + "1_person_gray.jpg")
uwb1 = np.load(dir_path + "1_person_UWB.npy")

img2 = mpimg.imread(dir_path + "2_person.jpg")
gray2 = mpimg.imread(dir_path + "2_person_gray.jpg")
uwb2 = np.load(dir_path + "2_person_UWB.npy")

fig, axs = plt.subplots(6, 1, figsize=(10, 16))

axs[0].imshow(img1)
axs[0].set_title("1_person.jpg")

axs[1].imshow(gray1, cmap="gray")
axs[1].set_title("1_person_gray.jpg")

axs[2].plot(uwb1)
axs[2].set_title("1_person_UWB.npy")
axs[2].set_xlim(0, len(uwb1)) # set x-axis limits to match image subplots

axs[3].imshow(img2)
axs[3].set_title("2_person.jpg")

axs[4].imshow(gray2, cmap="gray")
axs[4].set_title("2_person_gray.jpg")

axs[5].plot(uwb2)
axs[5].set_title("2_person_UWB.npy")
axs[5].set_xlim(0, len(uwb2)) # set x-axis limits to match image subplots

plt.tight_layout()
plt.show()
