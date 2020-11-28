from 共振峰估计函数 import *
from scipy.signal import lfilter
import librosa
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 12))
path="F:\\python\\VowelStuday\\SingleVowel\\aoeiu元音音频\\a不同人发音\\a2.wav"
#path="C4_3_y.wav"
#data, fs = soundBase('C4_3_y.wav').audioread()
data, fs = librosa.load(path, sr=None, mono=False)#sr=None声音保持原采样频率， mono=False声音保持原通道数
# 预处理-预加重
u = lfilter([1, -0.99], [1], data)

cepstL = 7
wlen = len(u)
wlen2 = wlen // 2
print("帧长={}".format(wlen))
print("帧移={}".format(wlen2))
# wlen = 256
# wlen2 = 256//2
# 预处理-加窗
u2 = np.multiply(u, np.hamming(wlen))
# 预处理-FFT,取对数 获得频域图像 取一半
U_abs = np.log(np.abs(np.fft.fft(u2))[:wlen2])
# 4.3.1
freq = [i * fs / wlen for i in range(wlen2)]
#print(freq)
#val共振峰幅值 loc共振峰位置 spec包络线
val, loc, spec = Formant_Cepst(u, cepstL)
print(int(freq[loc[0]]))
print(int(freq[loc[1]]))
plt.subplot(2, 1, 1)
plt.plot(freq, U_abs, 'k')
plt.xlabel('频率/Hz')           #设置x，y轴的标签
plt.ylabel('幅值')
plt.title('男性a的发音频谱')
plt.subplot(2, 1, 2)
plt.plot(freq, spec, 'k')
plt.xlabel('频率/Hz')           #设置x，y轴的标签
plt.ylabel('幅值')
plt.title('倒谱法共振峰估计')

for i in range(len(loc)):
    plt.subplot(2, 1, 2)
    plt.plot([freq[loc[i]], freq[loc[i]]], [np.min(spec), spec[loc[i]]], '-.k')
    plt.text(freq[loc[i]], spec[loc[i]], 'Freq={}'.format(int(freq[loc[i]])))

plt.savefig('images/共振峰估计.png')
plt.show()
plt.close()