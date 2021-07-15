#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ch05.04，page 135 误差反向传播法乘法层（mullayer）和加法层（addlayer）的实现
import sys, os
sys.path.append(os.pardir) #将父目录（上一级）os.pardir添加到系统搜索路径
#print(os.pardir) # 输出‘..’
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d
import functions_methods as fm
from dataset.mnist import load_mnist #在上一级中搜索dataset文件夹（模块）并得到其中mnist.py文件
from PIL import Image


#初始化
#1.乘法层
class MulLayer:
    
    def __init__(self):
        #初始化参数
        self.x=None
        self.y=None
        
    def forward(self, x, y):
        self.x=x            #存储过程x、y
        self.y=y
        out=x*y
        return y
    
    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy
    
#2.加法层    
class AddLayer:
    
    def __init__(self):
        pass
    
    def forward(self,x,y):
        out=x+y
        return out
    
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy
    
#3.ReLu激活函数层, 此时x为数组
class ReLU_layer:
    
    def __init__(self):
        self.mask=None 
    
    def forward(self, x):
        self.mask=(x<=0)  #存储被阻断的信号下标
        out=x.copy()
        out[self.mask]=0 #对于被阻断的信号，out即y=0
        return out
    
    def backward(self, dout):
        dout[self.mask]=0
        dx=dout
        #print('Through ReLU layer')
        return dx
    
#4.Sigmoid激活函数层
class Sigmoid_layer:
    
    def __init__(self, x):
        self.out=None
    
    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out
    
    def backward(self,dout):
        dx=dout*(1-self.out)*self.out
        #print('Through Sigmoid layer')
        return dx

#5.Affine仿射变换函数层
class Affine_layer:
    
    def __init__(self, w, b):
        self.w=w
        self.b=b
        self.x=None
        self.dw=None
        self.db=None
        
        
    def forward(self, x):
        self.x=x                      
        out=np.dot(x,self.w)+self.b
        return out
    
    def backward(self, dout):
        dx=np.dot(dout,self.w.T)
        self.dw=np.dot(self.x.T, dout)  #存储更新的参数？
        self.db=np.sum(dout,axis=0)
        #print('Through Affine layer')
        return dx

#6.softmax输出层（将得分正规化，输出概率）激活函数+Loss损失函数（cross entropy error）层
class SoftmaxWithLoss:
    
    def __init__(self):
        self.y=None
        self.t=None
        self.loss=None
    
    def forward(self, x, t):
        self.t=t
        self.y=fm.softmax(x)
        self.loss=fm.cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size
        #print('Through softmaxwithloss layer')
        return dx
    
        
try:    
    get_ipython().system('jupyter nbconvert --to python layers_error_back_propogation.ipynb')
    # python即转化为.py，script即转化为.html
except:
    pass


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




