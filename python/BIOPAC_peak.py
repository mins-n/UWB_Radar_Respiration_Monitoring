from util import Peak_Detection
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


# Load the .mat file
dir_path = "./../Data/2022.12.28/2022.12.28_1_gon_sun/2022.12.28_1_gon_sun.mat"
data = scipy.io.loadmat(dir_path)
data = np.array(data['channels'])  # Convert the data to a NumPy array

data1 = data[0][0][0][0][0][0]
data1_fs = data[0][0][0][0][1][0]
data1 = data1.flatten()  # Flatten the data if it is not 1D
data1 = data1.astype(float)  # Cast to float if necessary

data2 = data[0][1][0][0][0][0]
data2_fs = data[0][1][0][0][1][0]
data2 = data2.flatten()  # Flatten the data if it is not 1D
data2 = data2.astype(float)  # Cast to float if necessary

# Set the sample rate and minimum interval
MI = 1  # Minimum interval in seconds

data1_rpeak_i, data1_env_peak_i = Peak_Detection.Peak_Detection(data1, data1_fs, MI)
data2_rpeak_i, data2_env_peak_i = Peak_Detection.Peak_Detection(data2, data2_fs, MI)

# Print the detected peaks
print(data1_rpeak_i)
print(data2_rpeak_i)

plt.figure(num=1, figsize=(10, 20))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=1.3)
plt.subplot(2, 1, 1)
plt.subplot(2, 1, 1).set_title("BIOPAC Peak Detection Data1")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(data1)
data1_rpeak_i = data1_rpeak_i.astype(int)
plt.plot(data1_rpeak_i, data1[data1_rpeak_i], 'ro')

plt.subplot(2, 1, 2)
plt.subplot(2, 1, 2).set_title("BIOPAC Peak Detection Data2")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(data2)
data2_rpeak_i = data2_rpeak_i.astype(int)
plt.plot(data2_rpeak_i, data2[data2_rpeak_i], 'ro')
plt.show()