from 共振峰估计函数 import *
from scipy.signal import lfilter
import librosa
import numpy as np
import matplotlib.pyplot as plt

#plt.figure(figsize=(14, 12))
plt.xlim(100,1200) #x坐标轴范围-10~10
plt.ylim(800,2600)
x=[1021,596,469,372,508]
y=[2428,1583,996,1149,1543]
plt.plot(x,y,'ro',color='red')
plt.xlabel('第一共振峰频率/Hz')           #设置x，y轴的标签
plt.ylabel('第二共振峰频率/Hz')
#第一个参数为标记文本，第二个参数为标记对象的坐标，第三个参数为标记位置
plt.annotate('a', xy=(1021,2428), xytext=(1042,2442))
plt.annotate('o', xy=(596,1583), xytext=(596+20,1583+20))
plt.annotate('e', xy=(496,996), xytext=(496+20-9,996+20-9))
plt.annotate('i', xy=(372,1149), xytext=(372+20-6,1149+20-6))
plt.annotate('u', xy=(508,1543), xytext=(508+20-6,1543+20-6))

plt.grid()#网格线显示

plt.savefig('images/五元音共振峰.png')
plt.show()
plt.close()

