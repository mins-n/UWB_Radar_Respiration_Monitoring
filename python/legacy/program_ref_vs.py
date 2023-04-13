import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("./respiration_rate_predict")

dir_path = "./../Data/2023.01.10/2023.01.10_2_soo_gu/1/"
Biopack_path = dir_path + "data1_ref.npy"
UWB_input_path = dir_path + "1_person_gray.npy"
BIOPAC_data = np.load(Biopack_path,allow_pickle=True)


res_predict = []
data_list = []

tmp = np.load(UWB_input_path)
for i in range(110):
    resized_img = cv2.resize(tmp[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    data_list.append(resized_img)
data_list = np.array(data_list)
data_list = data_list.reshape(len(data_list), 28, 28, 1)
data_list = data_list / 255.0


res_predict = model.predict(data_list)

interval = 10
res_predict_averages = []
for i in range(0, len(res_predict), interval):
    slice_arr = res_predict[i:i+interval]
    avg = np.mean(slice_arr)
    res_predict_averages.append(avg)

BIOPAC_average = []
for i in range(0, len(BIOPAC_data), interval):
    slice_arr = BIOPAC_data[i:i+interval]
    avg = np.mean(slice_arr)
    BIOPAC_average.append(avg)


rsp_predict = []
rsp_ref = []
for res in res_predict_averages:
    rsp_predict = rsp_predict + nk.rsp_simulate(duration=10, sampling_rate=500, respiratory_rate=res).tolist()

for res in BIOPAC_average:
    rsp_ref = rsp_ref + nk.rsp_simulate(duration=10, sampling_rate=500, respiratory_rate=res).tolist()

plt.figure(num=1, figsize=(20, 14))
plt.title("Plot the respiratory rate of CNN results")
plt.plot(rsp_predict)
plt.plot(rsp_ref)
plt.xlabel('Time, 500fs')
plt.ylabel('Amplitude')
plt.legend(["CNN respiratory rate","BIOPAC"])
plt.show()
