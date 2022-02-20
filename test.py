#绘制正弦曲线

#引入数值计算库,改为短名称
import numpy as np
#引入绘图库，改为短名称
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('TkAgg')

#生成一个由-4到4、均分为200个元素的列表
x = np.linspace(-4, 4, 200) 
#计算当x取值范围-4至4时所有的sin函数解
f = np.sin(x)

#绘制
plt.plot(x, f, 'red') 

#将绘制好的图显示出来
plt.show()
