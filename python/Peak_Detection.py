import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def Peak_Detection(data, fs, MI):
    segLen = int(MI * fs / 4)
    Nseg = int(len(data) / segLen)
    segMaxIdx = np.array([])
    envPeakIdx = np.zeros(Nseg)
    interval_th = 0.4 * MI * fs
    minInterval = MI / 4
    maxInterval = MI * 2
    maxCnt = 1
    peakCnt = 1
    peakChk = 0
    curIdx = 0

    while curIdx <= len(data):
        sIdx = curIdx
        eIdx = curIdx + segLen - 1

        if eIdx > len(data):
            eIdx = len(data)
        tmpdata = data[sIdx:eIdx]

        m = np.max(tmpdata)
        i = np.argmax(tmpdata)
        segMaxIdx = np.append(segMaxIdx, sIdx + i)

        if maxCnt > 4:
            tmpIdx = segMaxIdx[maxCnt - 4:maxCnt - 1].astype(int)
            tmpdata2 = data[tmpIdx]
            ddata = np.diff(tmpdata2)

            if ddata[0] > 0 and ddata[1] < 0:
                envPeakIdx[peakCnt - 1] = tmpIdx[1]
                peakCnt += 1
                peakChk = 1

            if peakCnt > 3 and peakChk == 1:
                peakChk = 0
                tmpIdx = envPeakIdx[peakCnt - 4:peakCnt - 1].astype(int)
                tmpdata2 = data[tmpIdx]
                dIdx = np.diff(tmpIdx)
                fp_idx = np.where(dIdx < interval_th)[0]

                if len(fp_idx) > 0:
                    m = np.min(tmpdata2[fp_idx:fp_idx + 1])
                    k = np.argmin(tmpdata2[fp_idx:fp_idx + 1])
                    tmpIdx = tmpIdx[:fp_idx + k-1] + tmpIdx[fp_idx + k:]
                    tmpIdx.append(0)
                    envPeakIdx[peakCnt - 4:peakCnt - 1] = tmpIdx
                    peakCnt -= 1

                if peakCnt > 20:
                    tmpIdx = envPeakIdx[peakCnt - 11:peakCnt - 1].astype(int)
                    tmpdata2 = data[tmpIdx]
                    m_amp = np.median(tmpdata2)
                    abnormal = np.where((tmpdata2 < (m_amp / 3)) | (tmpdata2 > (m_amp * 3)))[0].astype(int)
                    if len(abnormal) > 0:
                        tmpIdx = np.array([x for i, x in enumerate(tmpIdx) if i not in abnormal])
                        nAbnormal = len(abnormal)
                        tmpIdx = np.resize(tmpIdx, tmpIdx.shape[0] + nAbnormal)
                        envPeakIdx[peakCnt - 11: peakCnt - 1] = tmpIdx
                        peakCnt -= nAbnormal
                        tmpIdx = envPeakIdx[peakCnt - 11 + nAbnormal: peakCnt - 1]

                    d_tmpIdx = np.diff((tmpIdx) / fs)
                    ab_interval = np.where((d_tmpIdx < minInterval) | (d_tmpIdx > maxInterval))[0]

                    if len(ab_interval) > 0:
                        tmpIdx = np.array([x for i, x in enumerate(tmpIdx) if i not in abnormal + 1])
                        d_tmpIdx = np.array([x for i, x in enumerate(d_tmpIdx) if i not in ab_interval])

                    if len(d_tmpIdx) > 0:
                        tmpMI = np.mean(d_tmpIdx)
                        if tmpMI < minInterval or tmpMI > maxInterval:
                            segLen = int(MI * fs / 4)
                        else:
                            segLen = int(tmpMI * fs / 4) - 1
        curIdx = eIdx + 1
        maxCnt = maxCnt + 1

    envPeakIdx = envPeakIdx[:peakCnt]
    peak_i = envPeakIdx
    return peak_i, segMaxIdx

# Load the .mat file
data = scipy.io.loadmat('data.mat')
data = np.array(data['data'])  # Convert the data to a NumPy array

data = data.flatten()  # Flatten the data if it is not 1D
data = data.astype(float)  # Cast to float if necessary

# Set the sample rate and minimum interval
fs = 125  # Sample rate in Hz
MI = 1  # Minimum interval in seconds

# Call the peak detection function
rpeak_i, env_peak_i = Peak_Detection(data, fs, MI)

# Print the detected peaks
print(rpeak_i)


plt.plot(data)
rpeak_i = rpeak_i.astype(int)
plt.plot(rpeak_i, data[rpeak_i], 'ro')
plt.show()