#!/usr/bin/env python
# coding: utf-8

# In[10]:


#ch03.02 page 42-50 (hidden layer's activation function)  
#ch03.05 page 63-69 (output layer's activation function)

get_ipython().run_line_magic('matplotlib', 'inline')
#表示在这里显示图像，不设置时只显示<Figure size 640x480 with 1 Axes>
import numpy as np
import matplotlib.pylab as plt
import matplotlib
#print(matplotlib.matplotlib_fname())

#step function
def step_function(x):# if x>=0:  return 1 elif x<0: return 0
    #y=np.array(x>0)
    y=np.array(x>0, dtype=np.int)         
    return y

def sigmoid(x):
    y=1/(1+np.exp(-x))  #和非数组计算的广播功能
    return y

#Rectified Linear Unit function
def ReLU(x):
  #  y=np.array(x)
  #  y[y<=0]=0.0
  #  return y
    return(np.maximum(x,0)) 


#输出层激活函数
#1. identity_function 恒等函数，apply to regression problems 适合回归问题
def identity(a):
    return a

#2. softmax_function ，apply to classification problems 适合分类问题
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)    #通过减去最大值防止因e指数后各个数值相差太大而出现的溢出现象。
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a  #sum_exp_a为一个常数，此处使用了广播功能
    return y           #sum(y)=1,且y的大小顺序同输入a(y=e**a为单调递增函数)，所以softmax层有时会被省略   

x=np.arange(-3, 3, 0.01)
y1=step_function(x)
y2=sigmoid(x)
y3=ReLU(x)
y4=softmax(x)

#plt.xlim(-3.5,3.5)
#plt.ylim(-1,3.5)
plt.xlabel('X')
plt.ylabel('Y')


plt.subplot(2,1,2)
plt.plot(x, y1, color='r', linewidth=1, linestyle='-',label='step_function')
plt.plot(x, y2, color='b', linewidth=1, linestyle='--',label='sigmoid_function')
plt.legend()
#plt.title('两者区别为连续和非连续')
#plt.tight_layout()

plt.subplot(2,2,2)
plt.plot(x, y3, color='c', linewidth=1.5, linestyle=':', label='ReLU_function')
plt.legend()
plt.title('ReLU')
#plt.tight_layout()

plt.subplot(2,2,1)
plt.plot(x, y4, color='m', linewidth=1.5, linestyle='-.', label='softmax_function')
plt.legend()
#plt.tight_layout()
#, x=0.5, y=1.05, ha='center', va='top'
Title=plt.suptitle('activation_function激活函数', fontsize=16)
Title.set_color('r')
#plt.tight_layout()
plt.savefig(fname='/Users/huangjing/Desktop/figures/activation_function.png', dpi=600)
plt.show()
#y=step_function(x)  #输出bool型
#y=y.astype(np.int) #bool型转换为int型时True为1，False为0

try:    
    get_ipython().system('jupyter nbconvert --to python activation_function.ipynb')
    # python即转化为.py，script即转化为.html
except:
    pass


# In[ ]:





# In[ ]:




